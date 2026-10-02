"""ConCor Video export: the mask tracks of every VidSTG relation in the record contract of
ConCor-Video-Data-Processing (schema `concor-video-tracklet-bcc-v2`), the pipeline our
output feeds.

A ConCor Video record is one text ↔ tracklet correspondence sample: the text, the list of
frame ids, one tracklet per instance with a mask on every frame (uncompressed COCO RLE, null
where the instance has no mask), `groups` (tracklet → text spans, with a role) and
`span_links` (the deterministic inverse), and a disposition. Here one record is one VidSTG
relation (subject, predicate, object): two tracklets, `vidor-<tid>`, built from our mask
records; the frames are the relation's segment; the text is the BCC-complete caption of the
relation when one is given (`--captions`), and a record without a caption is written to the
tracklets table only, because the schema requires a text and a main-referent group.

Mapping (docs/CONCOR_VIDEO.md has the full table and the open points):

  sample_id            vidstg:<split>:<vid>:<subject_tid>-<predicate>-<object_tid>
  dataset / cohort     "vidstg" / "relation"      (dataset is not in their enum yet)
  split                the VidSTG split (train / val / test)
  video_id             <vid>; expression_id = <subject_tid>-<predicate>-<object_tid>
  frame_ids            every frame of the relation segment, "%06d" (native frame index)
  frame_files          "<vid>/%06d.jpg" (the convention; no frames are written)
  tracklets[].source   "sam3.1_main_referent" for both objects (their enum has no value for
                       SAM masks prompted from ground-truth boxes; the box origin is kept in
                       source_annotation_id "vidor:<vid>:<tid>" and in `vidstg_provenance`)
  tracklets[].confidence   mean of mask_confidence over the masked frames (their mean_score)
  tracklets[].masks    aligned to frame_ids; null outside the object's box span or on a refusal
  groups               subject -> role main_referent, object -> role context_entity
  disposition          complete_bcc when both objects have a mask and a span; otherwise
                       missing_main_referent (subject without any mask) or incomplete_context

The record validator (`validate_record`, `rebuild_span_links`, `make_span`) and the
uncompressed-RLE conversion follow ConCor-Video-Data-Processing (MIT, suryathecreator), so
our records fail the same checks theirs would.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa

from .anchors import in_span_fids, plan_clip
from .datasets import load_vidor
from .export import _Sink
from .records import read_jsonl, write_json_atomic
from .worker import ledger, load_worklist, unit_records

SCHEMA_VERSION = "concor-video-tracklet-bcc-v2"
DATASET = "vidstg"
COHORT = "relation"
SOURCE = "sam3.1_main_referent"
SOURCES = ("ground_truth", "sam3.1_main_referent", "sam3.1_context")
DISPOSITIONS = ("complete_bcc", "incomplete_context", "missing_main_referent", "negative_unsegmentable")

# Their Parquet layouts (exporter.py of ConCor-Video-Data-Processing), column for column.
SAMPLE_SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("dataset", pa.string()), ("split", pa.string()), ("cohort", pa.string()),
    ("annotation_protocol", pa.string()), ("provenance_warning", pa.string()), ("dataset_root", pa.string()),
    ("frame_source", pa.string()), ("video_id", pa.string()), ("expression_id", pa.string()),
    ("text", pa.string()), ("negative", pa.bool_()), ("target_source", pa.string()),
    ("disposition", pa.string()), ("frame_count", pa.int32()), ("tracklet_count", pa.int32()),
    ("unresolved_context_count", pa.int32()), ("frame_ids_json", pa.string()),
    ("frame_files_json", pa.string()), ("span_links_json", pa.string()), ("extraction_json", pa.string()),
    ("sam_prompt_audit_json", pa.string()), ("pipeline_json", pa.string()), ("runtime_seconds", pa.float64()),
])
VERIFICATION_SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("dataset", pa.string()), ("split", pa.string()), ("cohort", pa.string()),
    ("annotation_protocol", pa.string()), ("provenance_warning", pa.string()), ("dataset_root", pa.string()),
    ("frame_source", pa.string()), ("video_id", pa.string()), ("expression_id", pa.string()),
    ("text", pa.string()), ("negative", pa.bool_()), ("target_source", pa.string()),
    ("disposition", pa.string()), ("frame_ids_json", pa.string()), ("frame_files_json", pa.string()),
    ("tracklets_json", pa.string()), ("groups_json", pa.string()), ("span_links_json", pa.string()),
    ("extraction_json", pa.string()), ("sam_prompt_audit_json", pa.string()), ("runtime_seconds", pa.float64()),
])
TRACKLET_SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("dataset", pa.string()), ("split", pa.string()), ("cohort", pa.string()),
    ("video_id", pa.string()), ("expression_id", pa.string()), ("text", pa.string()),
    ("disposition", pa.string()), ("tracklet_id", pa.string()), ("role", pa.string()),
    ("identity", pa.string()), ("source", pa.string()), ("source_annotation_id", pa.string()),
    ("sam_prompt", pa.string()), ("confidence", pa.float64()), ("max_confidence", pa.float64()),
    ("present_frames", pa.int32()), ("frame_count", pa.int32()), ("frame_ids_json", pa.string()),
    ("frame_files_json", pa.string()), ("text_spans_json", pa.string()), ("masks_rle_json", pa.string()),
])
LINK_SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("text", pa.string()), ("span_start", pa.int32()),
    ("span_end", pa.int32()), ("span_text", pa.string()), ("tracklet_ids_json", pa.string()),
])


# ── RLE: compressed COCO counts -> uncompressed counts (no pixel decode) ──────────────────

def compressed_counts(value: str | bytes) -> list[int]:
    """COCO's ASCII-compressed counts -> the run lengths (the LEB128-like scheme of
    pycocotools' rleFrString, as in ConCor-Video-Data-Processing rle.py)."""
    if isinstance(value, bytes):
        value = value.decode("ascii")
    counts: list[int] = []
    position = 0
    while position < len(value):
        number = 0
        shift = 0
        more = True
        while more:
            code = ord(value[position]) - 48
            position += 1
            number |= (code & 0x1F) << (5 * shift)
            more = bool(code & 0x20)
            if not more and code & 0x10:
                number |= -1 << (5 * (shift + 1))
            shift += 1
        if len(counts) > 2:
            number += counts[-2]
        counts.append(number)
    return counts


def uncompressed_rle(rle: dict) -> dict:
    """{size, counts: str} -> {size, counts: [int, ...]} whose runs sum to H x W."""
    h, w = int(rle["size"][0]), int(rle["size"][1])
    counts = rle["counts"]
    runs = compressed_counts(counts) if isinstance(counts, (str, bytes)) else [int(c) for c in counts]
    if any(r < 0 for r in runs) or sum(runs) != h * w:
        raise ValueError(f"RLE runs cover {sum(runs)} pixels, expected {h * w}")
    return {"size": [h, w], "counts": runs}


# ── the BCC record rules (as in their tracklet_schema.py) ───────────────────────────────

def make_span(text: str, start: int, end: int) -> dict:
    if not (0 <= start < end <= len(text)):
        raise ValueError(f"invalid span [{start}, {end}) for text of length {len(text)}")
    return {"start": start, "end": end, "text": text[start:end]}


def rebuild_span_links(groups: list[dict]) -> list[dict]:
    by_span: dict[tuple[int, int, str], set[str]] = {}
    for group in groups:
        ids = {str(v) for v in group.get("tracklet_ids", [])}
        for span in group.get("text_spans", []):
            by_span.setdefault((int(span["start"]), int(span["end"]), str(span["text"])), set()).update(ids)
    return [{"start": s, "end": e, "text": t, "tracklet_ids": sorted(ids)}
            for (s, e, t), ids in sorted(by_span.items())]


def validate_record(record: dict) -> None:
    """Fail closed on broken BCC links or truncated tracklets; the checks of
    ConCor-Video-Data-Processing plus their enums for source and disposition."""
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected schema_version")
    text = str(record.get("text", ""))
    if not text:
        raise ValueError("record has no text")
    frame_ids = list(record.get("frame_ids", []))
    if not frame_ids:
        raise ValueError("record has no frames")
    frame_files = list(record.get("frame_files", []))
    if frame_files and len(frame_files) != len(frame_ids):
        raise ValueError("frame_files must align one-to-one with frame_ids")
    by_id: dict[str, dict] = {}
    for t in record.get("tracklets", []):
        tid = str(t["tracklet_id"])
        if tid in by_id:
            raise ValueError(f"duplicate tracklet_id: {tid}")
        if t.get("source") not in SOURCES:
            raise ValueError(f"tracklet {tid}: source {t.get('source')!r} not in {SOURCES}")
        masks = list(t.get("masks", []))
        if len(masks) != len(frame_ids):
            raise ValueError(f"tracklet {tid} has {len(masks)} masks for {len(frame_ids)} frames")
        present = 0
        for rle in masks:
            if rle is None:
                continue
            present += 1
            if len(rle.get("size", [])) != 2 or any(int(v) <= 0 for v in rle["size"]):
                raise ValueError(f"tracklet {tid} has an invalid RLE size")
            counts = rle.get("counts")
            if not isinstance(counts, list) or sum(int(v) for v in counts) != int(rle["size"][0]) * int(rle["size"][1]):
                raise ValueError(f"tracklet {tid} has invalid uncompressed RLE counts")
        if int(t.get("present_frames", -1)) != present:
            raise ValueError(f"tracklet {tid}: present_frames {t.get('present_frames')} != {present} masks")
        by_id[tid] = t
    linked: set[str] = set()
    for group in record.get("groups", []):
        ids = [str(v) for v in group.get("tracklet_ids", [])]
        if not ids:
            raise ValueError("BCC group has no tracklet_ids")
        for tid in ids:
            if tid not in by_id:
                raise ValueError(f"group references missing tracklet {tid}")
            linked.add(tid)
        if not group.get("text_spans"):
            raise ValueError("BCC group has no text span")
        for span in group["text_spans"]:
            s, e = int(span["start"]), int(span["end"])
            if text[s:e] != span["text"]:
                raise ValueError(f"span mismatch [{s}, {e}): {span['text']!r} != {text[s:e]!r}")
    if linked != set(by_id):
        raise ValueError(f"unlinked retained tracklets: {sorted(set(by_id) - linked)}")
    if record.get("span_links") != rebuild_span_links(record.get("groups", [])):
        raise ValueError("span_links is not the canonical inverse of groups")
    disposition = record.get("disposition")
    if disposition not in DISPOSITIONS:
        raise ValueError(f"disposition {disposition!r} not in {DISPOSITIONS}")
    if record.get("negative") and (by_id or record.get("groups")):
        raise ValueError("negative record must not contain tracklets")
    if record.get("negative") and disposition != "negative_unsegmentable":
        raise ValueError("negative record has the wrong disposition")
    if not record.get("negative") and not by_id and disposition != "missing_main_referent":
        raise ValueError("positive record without tracklets must be a missing main referent")
    if by_id and not any(g.get("role") == "main_referent" for g in record["groups"]):
        raise ValueError("positive tracklets require a main_referent group")


# ── building records from a campaign ─────────────────────────────────────────────────────

def frame_id(fid: int) -> str:
    return f"{int(fid):06d}"


def tracklet_from_records(plan: dict, tid: int, recs: dict[int, dict], fids: list[int]) -> dict | None:
    """One ConCor tracklet for a VidOR object over `fids` (the relation's frames); None when
    the object has no mask on any of them. Masks are null outside the object's boxed span
    and on refusals; `vidstg_provenance` carries our per-record provenance summary."""
    masks, confs, refusals = [], [], Counter()
    anchors: list[int] = []
    prov: dict = {}
    for f in fids:
        r = recs.get(f)
        if r is None or r.get("rle") is None:
            masks.append(None)
            if r is not None:
                refusals[r.get("reason_code") or "unknown"] += 1
            continue
        masks.append(uncompressed_rle(r["rle"]))
        if r.get("mask_confidence") is not None:
            confs.append(float(r["mask_confidence"]))
        if not prov:
            pp = r.get("prompt_payload") or {}
            anchors = [int(a) for a in (pp.get("anchor_fids") or [])]
            prov = dict(model=r.get("model"), checkpoint_hash=r.get("checkpoint_hash"),
                        sam_version=r.get("sam_version"), prompt_mode=r.get("prompt_mode"),
                        anchor_policy=pp.get("anchor_policy"), code_commit=r.get("code_commit"))
    present = sum(1 for m in masks if m is not None)
    if present == 0:
        return None
    span = plan["spans"][tid]
    return {
        "tracklet_id": f"vidor-{tid}",
        "source": SOURCE,
        "source_annotation_id": f"vidor:{plan['vid']}:{tid}",
        "sam_prompt": f"VidOR box of tid {tid} at {len(anchors)} keyframes"
                      + (f" ({prov['anchor_policy']})" if prov.get("anchor_policy") else ""),
        "confidence": (sum(confs) / len(confs)) if confs else 1.0,
        "max_confidence": max(confs) if confs else 1.0,
        "present_frames": present,
        "masks": masks,
        "vidstg_provenance": dict(prov, tid=tid, category=plan["cats"].get(tid),
                                  span=[int(span[0]), int(span[1])], anchor_fids=anchors,
                                  frames_with_box=len(recs), refusals=dict(refusals)),
    }


def relation_record(plan: dict, split: str, segment: tuple[int, int], relation: tuple,
                    records: dict[int, dict[int, dict]], caption: dict | None = None) -> dict:
    """The ConCor record of one relation. `records[tid][fid]` are our mask records;
    `caption` is {"text": ..., "spans": {"<tid>": [[start, end], ...]}} or None."""
    s, p, o = relation
    vid = plan["vid"]
    fids = list(range(int(segment[0]), int(segment[1]) + 1))
    expression_id = f"{s}-{p}-{o}"
    rec = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"{DATASET}:{split}:{vid}:{expression_id}",
        "dataset": DATASET, "split": split, "cohort": COHORT,
        "annotation_protocol": "vidor_boxes_vidstg_relation",
        "provenance_warning": "masks are SAM 3.1 predictions prompted from VidOR ground-truth boxes",
        "video_id": vid, "expression_id": expression_id,
        "text": (caption or {}).get("text", ""),
        "negative": False,
        "frame_ids": [frame_id(f) for f in fids],
        "frame_files": [f"{vid}/{frame_id(f)}.jpg" for f in fids],
        "tracklets": [], "groups": [], "span_links": [],
        "extraction": {"relation": {"subject_tid": s, "predicate": p, "object_tid": o,
                                    "subject": plan["cats"].get(s), "object": plan["cats"].get(o)}},
        "sam_prompt_audit": [],
        "pipeline": {"target_source": "sam3.1_vidor_box_prompt", "context_source": "sam3.1_vidor_box_prompt",
                     "mask_records": "vidstg-masks records/<vid>.jsonl"},
        "vidstg_refused_tracklets": [],
    }
    roles = {s: "main_referent", o: "context_entity"}
    tracklets_by_tid: dict[int, dict] = {}
    for tid in (s, o) if s != o else (s,):
        t = tracklet_from_records(plan, tid, records.get(tid, {}), fids)
        rec["sam_prompt_audit"].append({
            "role": roles[tid], "sam_prompt": f"VidOR box of tid {tid}",
            "surface_spans": [plan["cats"].get(tid)], "raw_tracklets": 1,
            "retained_tracklets": int(t is not None),
            "rejections": [] if t is not None else [{"sam_object_id": tid, "reason": "no_mask_on_any_frame"}],
        })
        if t is None:
            rec["vidstg_refused_tracklets"].append({"tid": tid, "role": roles[tid]})
            continue
        rec["tracklets"].append(t)
        tracklets_by_tid[tid] = t
    if caption is None:
        rec["disposition"] = ""      # not a ConCor record yet: every tracklet kept for the tables
        return rec
    if caption:
        spans = {int(k): v for k, v in (caption.get("spans") or {}).items()}
        for tid, t in tracklets_by_tid.items():
            sp = [make_span(rec["text"], int(a), int(b)) for a, b in spans.get(tid, [])]
            if sp:
                rec["groups"].append({"group_id": f"vidor-{tid}", "role": roles[tid],
                                      "identity": plan["cats"].get(tid), "text_spans": sp,
                                      "tracklet_ids": [t["tracklet_id"]]})
        rec["span_links"] = rebuild_span_links(rec["groups"])
    if s not in tracklets_by_tid:
        rec["disposition"] = "missing_main_referent"
        # a tracklet without a span cannot stay in a positive record (their validator)
        rec["tracklets"] = [t for t in rec["tracklets"] if any(t["tracklet_id"] in g["tracklet_ids"] for g in rec["groups"])]
        if not rec["groups"]:
            rec["tracklets"] = []
    elif caption and o in tracklets_by_tid and all(
            any(t["tracklet_id"] in g["tracklet_ids"] for g in rec["groups"]) for t in rec["tracklets"]):
        rec["disposition"] = "complete_bcc"
    else:
        rec["disposition"] = "incomplete_context"
        rec["tracklets"] = [t for t in rec["tracklets"] if any(t["tracklet_id"] in g["tracklet_ids"] for g in rec["groups"])]
        if not any(g["role"] == "main_referent" for g in rec["groups"]):
            rec["tracklets"], rec["disposition"] = [], "missing_main_referent"
    return rec


def read_captions(path: Path | None) -> dict[tuple, dict]:
    """captions.jsonl rows: {vid, subject_tid, predicate, object_tid, text, spans: {"<tid>":
    [[start, end], ...]}} -> {(vid, s, p, o): row}."""
    out: dict[tuple, dict] = {}
    if path is None:
        return out
    for row in read_jsonl(path):
        out[(str(row["vid"]), int(row["subject_tid"]), str(row["predicate"]), int(row["object_tid"]))] = row
    return out


def _tracklet_row(rec: dict, t: dict | None, tid: int, role: str, identity: str | None) -> dict:
    spans = [sp for g in rec["groups"] if t and t["tracklet_id"] in g["tracklet_ids"] for sp in g["text_spans"]]
    return dict(sample_id=rec["sample_id"], dataset=rec["dataset"], split=rec["split"], cohort=rec["cohort"],
                video_id=rec["video_id"], expression_id=rec["expression_id"], text=rec["text"],
                disposition=rec.get("disposition", ""), tracklet_id=f"vidor-{tid}", role=role,
                identity=identity or "", source=SOURCE, source_annotation_id=f"vidor:{rec['video_id']}:{tid}",
                sam_prompt=t["sam_prompt"] if t else None, confidence=t["confidence"] if t else None,
                max_confidence=t.get("max_confidence") if t else None,
                present_frames=t["present_frames"] if t else 0, frame_count=len(rec["frame_ids"]),
                frame_ids_json=json.dumps(rec["frame_ids"]), frame_files_json=json.dumps(rec["frame_files"]),
                text_spans_json=json.dumps(spans),
                masks_rle_json=json.dumps(t["masks"] if t else [None] * len(rec["frame_ids"])))


def export_concor(worklist_path: Path, campaign_root: Path, output_dir: Path | None = None,
                  captions_path: Path | None = None) -> dict:
    """Every done clip of a campaign -> <output_dir>/{records/<sample_id>.json (captioned
    relations only), samples.parquet, tracklets.parquet, links.parquet,
    verification.parquet, manifest.json}. Relations without a caption appear in
    tracklets.parquet with an empty text and disposition (the caption stage fills them)."""
    wl = load_worklist(worklist_path)
    out = output_dir or (campaign_root / "export" / "concor")
    (out / "records").mkdir(parents=True, exist_ok=True)
    captions = read_captions(captions_path)
    sinks = {name: _Sink(out / f"{name}.parquet", schema) for name, schema in
             (("samples", SAMPLE_SCHEMA), ("tracklets", TRACKLET_SCHEMA), ("links", LINK_SCHEMA),
              ("verification", VERIFICATION_SCHEMA))}
    n_rel = n_cap = n_valid = 0
    disp: Counter = Counter()
    problems: list[str] = []
    try:
        for unit, row in zip(wl["units"], ledger(wl, campaign_root)):
            if row["status"] != "done":
                continue
            ann_path = Path(unit["vidor_ann"])
            if not ann_path.is_file():
                problems.append(f"{unit['vid']}: VidOR annotation {ann_path} not found; clip skipped")
                continue
            plan = plan_clip(unit["vid"], unit_records(unit), load_vidor(ann_path), unit["vidstg_split"], 0, "human")
            records: dict[int, dict[int, dict]] = defaultdict(dict)
            for r in read_jsonl(campaign_root / "records" / f"{unit['vid']}.jsonl"):
                records[int(r["tid"])][int(r["fid"])] = r
            for s, p, o in unit["relations"]:
                n_rel += 1
                cap = captions.get((unit["vid"], int(s), str(p), int(o)))
                rec = relation_record(plan, unit["vidstg_split"], tuple(unit["segment"]), (int(s), str(p), int(o)),
                                      records, cap)
                roles = {int(s): "main_referent", int(o): "context_entity"}
                by_tid = {int(t["tracklet_id"].split("-")[1]): t for t in rec["tracklets"]}
                for tid in sorted(set((int(s), int(o)))):
                    sinks["tracklets"].append(_tracklet_row(rec, by_tid.get(tid), tid, roles[tid], plan["cats"].get(tid)))
                if cap is None:
                    continue
                n_cap += 1
                try:
                    validate_record(rec)
                except ValueError as e:
                    problems.append(f"{rec['sample_id']}: {e}")
                    continue
                n_valid += 1
                disp[rec["disposition"]] += 1
                write_json_atomic(out / "records" / f"{rec['sample_id'].replace(':', '_')}.json", rec)
                common = dict(sample_id=rec["sample_id"], dataset=rec["dataset"], split=rec["split"],
                              cohort=rec["cohort"], annotation_protocol=rec["annotation_protocol"],
                              provenance_warning=rec["provenance_warning"], dataset_root="",
                              frame_source="vidor_video", video_id=rec["video_id"],
                              expression_id=rec["expression_id"], text=rec["text"], negative=False,
                              target_source=SOURCE, disposition=rec["disposition"],
                              frame_ids_json=json.dumps(rec["frame_ids"]),
                              frame_files_json=json.dumps(rec["frame_files"]),
                              span_links_json=json.dumps(rec["span_links"]),
                              extraction_json=json.dumps(rec["extraction"]),
                              sam_prompt_audit_json=json.dumps(rec["sam_prompt_audit"]), runtime_seconds=0.0)
                sinks["samples"].append(dict(common, frame_count=len(rec["frame_ids"]),
                                             tracklet_count=len(rec["tracklets"]),
                                             unresolved_context_count=len(rec["vidstg_refused_tracklets"]),
                                             pipeline_json=json.dumps(rec["pipeline"])))
                sinks["verification"].append(dict(common, tracklets_json=json.dumps(rec["tracklets"]),
                                                  groups_json=json.dumps(rec["groups"])))
                for link in rec["span_links"]:
                    sinks["links"].append(dict(sample_id=rec["sample_id"], text=rec["text"],
                                               span_start=link["start"], span_end=link["end"],
                                               span_text=link["text"], tracklet_ids_json=json.dumps(link["tracklet_ids"])))
        for s_ in sinks.values():
            s_.commit()
    except BaseException:
        for s_ in sinks.values():
            s_.abort()
        raise
    manifest = dict(
        schema_version=SCHEMA_VERSION, worklist=str(worklist_path), campaign_root=str(campaign_root),
        captions=str(captions_path) if captions_path else None,
        exported_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        relations=n_rel, relations_with_caption=n_cap, records_written=n_valid,
        relations_without_caption=n_rel - n_cap, dispositions=dict(disp),
        tables={k: f"{k}.parquet" for k in sinks}, records_dir="records/",
        tracklet_rows=sinks["tracklets"].count, problems=problems[:200], ok=not problems,
        open_points=["dataset 'vidstg' and cohort 'relation' are not in the ConCor Video enums",
                     "source 'sam3.1_main_referent' stands for SAM masks prompted from VidOR boxes"],
    )
    write_json_atomic(out / "manifest.json", manifest)
    return manifest
