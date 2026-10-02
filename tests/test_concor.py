import json

import pyarrow.parquet as pq
import pytest

from conftest import VID_A, VID_B, VID_D
from test_worker import fake_runner
from vidstg_masks.concor import (SCHEMA_VERSION, compressed_counts, export_concor, rebuild_span_links,
                                 uncompressed_rle, validate_record)
from vidstg_masks.records import rle_decode, rle_encode
from vidstg_masks.worker import build_worklist, process, save_worklist
import numpy as np


def _campaign(roots, tmp_path):
    c = tmp_path / "campaign"
    save_worklist(c / "worklist.json", build_worklist(roots, "all"))
    assert process(c / "worklist.json", 0, 1, c, c / "ckpt.pt", roots=roots,
                   clip_runner=fake_runner(roots, c, [])) == 0
    return c


def _captions(tmp_path):
    rows = [
        # A, relation 0 hold 1: both objects masked -> complete_bcc
        dict(vid=VID_A, subject_tid=0, predicate="hold", object_tid=1,
             text="An adult holds a ball.", spans={"0": [[0, 8]], "1": [[15, 21]]}),
        # A, relation 0 watch 2: tid 2 is refused on every frame (no human keyframe) -> incomplete_context
        dict(vid=VID_A, subject_tid=0, predicate="watch", object_tid=2,
             text="An adult watches a dog.", spans={"0": [[0, 8]], "2": [[17, 22]]}),
        # D: the video is missing, every record is a refusal -> missing_main_referent
        dict(vid=VID_D, subject_tid=0, predicate="hold", object_tid=1,
             text="A child holds a toy.", spans={"0": [[0, 7]], "1": [[14, 19]]}),
    ]
    p = tmp_path / "captions.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_uncompressed_rle_round_trips_through_pycocotools():
    m = np.zeros((48, 64), bool)
    m[5:20, 7:30] = True
    m[0, 0] = True                      # a foreground first pixel needs a leading zero run
    rle = rle_encode(m)
    u = uncompressed_rle(rle)
    assert u["size"] == [48, 64] and sum(u["counts"]) == 48 * 64 and u["counts"][0] == 0
    assert compressed_counts(rle["counts"]) == u["counts"]
    # the runs describe the same pixels pycocotools decodes
    flat = np.zeros(48 * 64, bool)
    pos, fg = 0, False
    for run in u["counts"]:
        if fg:
            flat[pos:pos + run] = True
        pos += run
        fg = not fg
    assert np.array_equal(flat.reshape((48, 64), order="F"), rle_decode(rle))
    with pytest.raises(ValueError):
        uncompressed_rle({"size": [48, 64], "counts": [10, 10]})


def test_export_concor_records_tables_and_dispositions(roots, tmp_path):
    c = _campaign(roots, tmp_path)
    manifest = export_concor(c / "worklist.json", c, captions_path=_captions(tmp_path))
    assert manifest["ok"], manifest["problems"]
    # A has 2 relations, B 1, D 1 (C is refused as too_many_objects but still done -> 16 relations)
    assert manifest["relations_with_caption"] == 3 and manifest["records_written"] == 3
    assert manifest["dispositions"] == {"complete_bcc": 1, "incomplete_context": 1, "missing_main_referent": 1}
    out = c / "export" / "concor"
    recs = {p.name: json.loads(p.read_text()) for p in (out / "records").glob("*.json")}
    assert len(recs) == 3
    hold = recs[f"vidstg_train_{VID_A}_0-hold-1.json"]
    validate_record(hold)
    assert hold["schema_version"] == SCHEMA_VERSION and hold["sample_id"] == f"vidstg:train:{VID_A}:0-hold-1"
    assert hold["frame_ids"] == [f"{f:06d}" for f in range(12)] and len(hold["frame_files"]) == 12
    assert [t["tracklet_id"] for t in hold["tracklets"]] == ["vidor-0", "vidor-1"]
    t1 = hold["tracklets"][1]
    assert t1["source"] == "sam3.1_main_referent" and t1["source_annotation_id"] == f"vidor:{VID_A}:1"
    assert len(t1["masks"]) == 12 and t1["present_frames"] == 10          # tid 1 is boxed on frames 2..11
    assert t1["masks"][0] is None and sum(t1["masks"][5]["counts"]) == 48 * 64
    assert t1["vidstg_provenance"]["category"] == "ball/sports_ball" and t1["vidstg_provenance"]["model"]
    assert [g["role"] for g in hold["groups"]] == ["main_referent", "context_entity"]
    assert hold["span_links"] == rebuild_span_links(hold["groups"])
    assert hold["span_links"][0] == {"start": 0, "end": 8, "text": "An adult", "tracklet_ids": ["vidor-0"]}
    watch = recs[f"vidstg_train_{VID_A}_0-watch-2.json"]
    validate_record(watch)
    assert watch["disposition"] == "incomplete_context" and [t["tracklet_id"] for t in watch["tracklets"]] == ["vidor-0"]
    assert watch["vidstg_refused_tracklets"] == [{"tid": 2, "role": "context_entity"}]
    missing = recs[f"vidstg_val_{VID_D}_0-hold-1.json"]
    validate_record(missing)
    assert missing["disposition"] == "missing_main_referent" and missing["tracklets"] == [] and missing["groups"] == []
    # their tables
    tr = pq.read_table(out / "tracklets.parquet").to_pylist()
    assert manifest["tracklet_rows"] == len(tr)
    b_rows = [r for r in tr if r["video_id"] == VID_B]
    assert len(b_rows) == 2 and all(r["text"] == "" and r["disposition"] == "" for r in b_rows)   # no caption yet
    assert all(len(json.loads(r["masks_rle_json"])) == r["frame_count"] for r in tr)
    a_hold = [r for r in tr if r["sample_id"] == hold["sample_id"]]
    assert {r["role"] for r in a_hold} == {"main_referent", "context_entity"}
    assert json.loads(a_hold[0]["text_spans_json"])
    assert len(pq.read_table(out / "samples.parquet")) == 3
    assert len(pq.read_table(out / "verification.parquet")) == 3
    links = pq.read_table(out / "links.parquet").to_pylist()
    assert len(links) == 3                                               # 2 spans in hold + 1 in watch
    assert (out / "manifest.json").is_file() and manifest["relations_without_caption"] == manifest["relations"] - 3


def test_validate_record_fails_closed(roots, tmp_path):
    c = _campaign(roots, tmp_path)
    export_concor(c / "worklist.json", c, captions_path=_captions(tmp_path))
    rec = json.loads((c / "export" / "concor" / "records" / f"vidstg_train_{VID_A}_0-hold-1.json").read_text())
    validate_record(rec)
    bad = json.loads(json.dumps(rec))
    bad["groups"][0]["text_spans"][0]["text"] = "An adults"
    with pytest.raises(ValueError, match="span mismatch"):
        validate_record(bad)
    bad = json.loads(json.dumps(rec))
    bad["tracklets"][0]["masks"].pop()
    with pytest.raises(ValueError, match="masks for"):
        validate_record(bad)
    bad = json.loads(json.dumps(rec))
    bad["span_links"] = []
    with pytest.raises(ValueError, match="canonical inverse"):
        validate_record(bad)
    bad = json.loads(json.dumps(rec))
    bad["tracklets"][0]["source"] = "sam3.1_vidor_box"
    with pytest.raises(ValueError, match="source"):
        validate_record(bad)


def test_export_concor_without_captions_writes_tracklets_only(roots, tmp_path):
    c = _campaign(roots, tmp_path)
    manifest = export_concor(c / "worklist.json", c)
    assert manifest["ok"] and manifest["records_written"] == 0 and manifest["relations_with_caption"] == 0
    assert manifest["tracklet_rows"] > 0 and not list((c / "export" / "concor" / "records").glob("*.json"))
