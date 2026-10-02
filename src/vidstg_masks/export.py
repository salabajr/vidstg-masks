"""records/*.jsonl -> export/{masks.parquet, refusals.parquet, manifest.json, ledger.csv}
plus an integrity check against the VidOR boxes.

Integrity: for every completed clip, every (tid, fid) that carries a VidOR box inside the
tid span has exactly one record; every RLE decodes to the annotation's HxW; every
provenance field is non-null. Problems are listed in manifest.json["integrity"] and make
the CLI exit non-zero; the tables are still written.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .anchors import in_span_fids, plan_clip
from .datasets import load_vidor
from .records import PROVENANCE_FIELDS, read_jsonl, rle_decode, validate_record, write_json_atomic
from .worker import ledger, load_worklist, unit_records

PROV_COLUMNS = [
    ("model", pa.string()), ("checkpoint_hash", pa.string()), ("sam_version", pa.string()),
    ("prompt_mode", pa.string()), ("prompt_payload", pa.string()),
    ("box_generated", pa.int8()), ("box_tracker", pa.string()), ("split", pa.string()),
    ("code_commit", pa.string()), ("created_at", pa.string()),
]
MASK_SCHEMA = pa.schema([("vid", pa.string()), ("tid", pa.int32()), ("fid", pa.int32()),
                         ("rle_size_h", pa.int32()), ("rle_size_w", pa.int32()),
                         ("rle_counts", pa.string()), ("mask_confidence", pa.float64())]
                        + PROV_COLUMNS)
REFUSAL_SCHEMA = pa.schema([("vid", pa.string()), ("tid", pa.int32()), ("fid", pa.int32()),
                            ("reason_code", pa.string())] + PROV_COLUMNS)


class _Sink:
    """Streaming Parquet writer to a .part file, renamed on commit."""

    def __init__(self, path: Path, schema: pa.Schema, batch_rows: int = 4096):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.schema, self.batch_rows = path, schema, batch_rows
        self.tmp = path.with_suffix(path.suffix + f".{os.getpid()}.part")
        self.writer = pq.ParquetWriter(self.tmp, schema, compression="zstd")
        self.rows: list[dict] = []
        self.count = 0

    def append(self, row: dict) -> None:
        self.rows.append(row)
        self.count += 1
        if len(self.rows) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows = []

    def commit(self) -> None:
        self.flush()
        self.writer.close()
        self.tmp.replace(self.path)

    def abort(self) -> None:
        try:
            self.writer.close()
        finally:
            self.tmp.unlink(missing_ok=True)


def _prov(rec: dict) -> dict:
    return dict(model=rec.get("model"), checkpoint_hash=rec.get("checkpoint_hash"),
                sam_version=rec.get("sam_version"), prompt_mode=rec.get("prompt_mode"),
                prompt_payload=json.dumps(rec.get("prompt_payload"), sort_keys=True),
                box_generated=rec.get("box_generated"), box_tracker=rec.get("box_tracker"),
                split=rec.get("split"), code_commit=rec.get("code_commit"), created_at=rec.get("created_at"))


def check_clip(unit: dict, records: list[dict], ann: dict | None, max_problems: int = 20) -> list[str]:
    """Integrity problems of one clip's records (empty list = clean)."""
    problems: list[str] = []
    seen: Counter = Counter()
    for i, rec in enumerate(records):
        try:
            validate_record(rec)
        except ValueError as e:
            problems.append(f"{unit['vid']} line {i + 1}: {e}")
            if len(problems) >= max_problems:
                return problems
            continue
        if rec["vid"] != unit["vid"]:
            problems.append(f"{unit['vid']} line {i + 1}: vid {rec['vid']} in the wrong file")
        seen[(rec["tid"], rec["fid"])] += 1
        if rec["rle"] is not None:
            try:
                m = rle_decode(rec["rle"])
            except Exception as e:  # noqa: BLE001
                problems.append(f"{unit['vid']} tid {rec['tid']} fid {rec['fid']}: rle does not decode ({e})")
                continue
            if ann is not None and m.shape != (ann["height"], ann["width"]):
                problems.append(f"{unit['vid']} tid {rec['tid']} fid {rec['fid']}: mask {m.shape} != "
                                f"annotation ({ann['height']}, {ann['width']})")
        if len(problems) >= max_problems:
            return problems
    dups = [k for k, n in seen.items() if n > 1]
    if dups:
        problems.append(f"{unit['vid']}: {len(dups)} duplicate (tid, fid) records, e.g. {dups[:3]}")
    if ann is not None:
        plan = plan_clip(unit["vid"], unit_records(unit), ann, unit["vidstg_split"], 0, "human")
        expected = {(t, f) for t in plan["tids"] for f in in_span_fids(plan, t)}
        missing = sorted(expected - set(seen))
        extra = sorted(set(seen) - expected)
        if missing:
            problems.append(f"{unit['vid']}: {len(missing)} boxed in-span (tid, fid) without a record, "
                            f"e.g. {missing[:3]}")
        if extra:
            problems.append(f"{unit['vid']}: {len(extra)} records outside the boxed span, e.g. {extra[:3]}")
    return problems


def export_campaign(worklist_path: Path, campaign_root: Path, output_dir: Path | None = None,
                    check_integrity: bool = True) -> dict:
    wl = load_worklist(worklist_path)
    out = output_dir or (campaign_root / "export")
    out.mkdir(parents=True, exist_ok=True)
    rows = ledger(wl, campaign_root)
    masks, refusals = _Sink(out / "masks.parquet", MASK_SCHEMA), _Sink(out / "refusals.parquet", REFUSAL_SCHEMA)
    by_reason, by_split_masks, by_split_refusals = Counter(), Counter(), Counter()
    commits, hashes, problems, clips_checked, invalid = set(), set(), [], 0, 0
    gpu_seconds, vram_max = 0.0, 0.0
    try:
        for unit, row in zip(wl["units"], rows):
            if row["status"] != "done":
                continue
            recs = list(read_jsonl(campaign_root / "records" / f"{unit['vid']}.jsonl"))
            ann = None
            if check_integrity:
                p = Path(unit["vidor_ann"])
                ann = load_vidor(p) if p.is_file() else None
                if ann is None:
                    problems.append(f"{unit['vid']}: VidOR annotation {p} not found; box coverage unchecked")
                problems.extend(check_clip(unit, recs, ann))
                clips_checked += 1
            for rec in recs:
                if check_integrity:
                    try:
                        validate_record(rec)
                    except ValueError:
                        invalid += 1      # reported by check_clip; never enters the tables
                        continue
                commits.add(rec.get("code_commit"))
                hashes.add(rec.get("checkpoint_hash"))
                base = dict(vid=rec["vid"], tid=rec["tid"], fid=rec["fid"], **_prov(rec))
                if rec.get("rle") is None:
                    refusals.append(dict(base, reason_code=rec.get("reason_code")))
                    by_reason[rec.get("reason_code")] += 1
                    by_split_refusals[rec["split"]] += 1
                else:
                    masks.append(dict(base, rle_size_h=rec["rle"]["size"][0], rle_size_w=rec["rle"]["size"][1],
                                      rle_counts=rec["rle"]["counts"], mask_confidence=rec["mask_confidence"]))
                    by_split_masks[rec["split"]] += 1
            run = ((rows and row.get("wall_s")) or 0) or 0
            gpu_seconds += float(run or 0)
            if row.get("vram_gb") not in ("", None):
                vram_max = max(vram_max, float(row["vram_gb"]))
        masks.commit()
        refusals.commit()
    except BaseException:
        masks.abort()
        refusals.abort()
        raise
    ledger_tmp = out / f"ledger.csv.{os.getpid()}.part"
    with open(ledger_tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["vid"])
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    ledger_tmp.replace(out / "ledger.csv")
    status = Counter(r["status"] for r in rows)
    manifest = dict(
        worklist=str(worklist_path), campaign_root=str(campaign_root), split=wl["split"],
        exported_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        clips=dict(status), masks=masks.count, refusals=refusals.count,
        refusals_by_reason=dict(sorted(by_reason.items(), key=lambda kv: str(kv[0]))),
        masks_by_split=dict(by_split_masks), refusals_by_split=dict(by_split_refusals),
        code_commits=sorted(c for c in commits if c), checkpoint_hashes=sorted(h for h in hashes if h),
        gpu_wall_hours=round(gpu_seconds / 3600, 3), vram_gb_max=round(vram_max, 2),
        tables={"masks": "masks.parquet", "refusals": "refusals.parquet", "ledger": "ledger.csv"},
        provenance_fields=list(PROVENANCE_FIELDS),
        integrity=dict(checked=check_integrity, clips_checked=clips_checked, ok=not problems,
                       invalid_records_excluded=invalid, problem_count=len(problems),
                       problems=problems[:200]),
    )
    write_json_atomic(out / "manifest.json", manifest)
    return manifest
