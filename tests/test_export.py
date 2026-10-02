import csv
import json

import pyarrow.parquet as pq

from conftest import VID_A, VID_B, VID_C, VID_D
from test_worker import fake_runner
from vidstg_masks.export import export_campaign
from vidstg_masks.records import PROVENANCE_FIELDS, rle_decode
from vidstg_masks.worker import build_worklist, process, save_worklist


def _run_campaign(roots, tmp_path):
    c = tmp_path / "campaign"
    save_worklist(c / "worklist.json", build_worklist(roots, "all"))
    assert process(c / "worklist.json", 0, 1, c, c / "ckpt.pt", roots=roots,
                   clip_runner=fake_runner(roots, c, [])) == 0
    return c


def test_export_integrity_on_a_complete_campaign(roots, tmp_path):
    c = _run_campaign(roots, tmp_path)
    manifest = export_campaign(c / "worklist.json", c)
    assert manifest["integrity"]["ok"], manifest["integrity"]["problems"]
    assert manifest["clips"] == {"done": 4}
    masks = pq.read_table(c / "export" / "masks.parquet").to_pylist()
    refusals = pq.read_table(c / "export" / "refusals.parquet").to_pylist()
    assert manifest["masks"] == len(masks) and manifest["refusals"] == len(refusals)
    # A: tids 0 and 1 masked on their boxed frames (12 + 10), tid 2 refused (12); B: 24 masks
    assert len(masks) == 22 + 24
    assert manifest["refusals_by_reason"] == {"no_human_keyframe": 12, "too_many_objects": 17 * 12,
                                              "video_missing": 24}
    assert manifest["masks_by_split"] == {"train": 22, "val": 24}
    assert "fake" in manifest["code_commits"] and all(len(h) == 64 for h in manifest["checkpoint_hashes"])
    row = masks[0]
    for f in PROVENANCE_FIELDS:
        if f == "rle":
            assert row["rle_counts"] and row["rle_size_h"] == 48 and row["rle_size_w"] == 64
        else:
            assert row[f] is not None
    assert rle_decode({"size": [row["rle_size_h"], row["rle_size_w"]], "counts": row["rle_counts"]}).all()
    assert json.loads(row["prompt_payload"])["anchor_policy"] == "human"
    with open(c / "export" / "ledger.csv") as f:
        rows = list(csv.DictReader(f))
    assert {r["vid"]: r["status"] for r in rows} == {v: "done" for v in (VID_A, VID_B, VID_C, VID_D)}
    assert (c / "export" / "manifest.json").is_file()


def test_export_reports_missing_and_duplicate_records(roots, tmp_path):
    c = _run_campaign(roots, tmp_path)
    p = c / "records" / f"{VID_A}.jsonl"
    lines = p.read_text().splitlines()
    dropped = json.loads(lines[3])
    p.write_text("\n".join(lines[:3] + lines[4:] + [lines[5]]) + "\n")     # one missing, one duplicated
    manifest = export_campaign(c / "worklist.json", c)
    assert not manifest["integrity"]["ok"]
    joined = " ".join(manifest["integrity"]["problems"])
    assert "duplicate" in joined and "without a record" in joined
    assert f"({dropped['tid']}, {dropped['fid']})" in joined


def test_export_reports_corrupt_provenance(roots, tmp_path):
    c = _run_campaign(roots, tmp_path)
    p = c / "records" / f"{VID_B}.jsonl"
    recs = [json.loads(l) for l in p.read_text().splitlines()]
    del recs[0]["checkpoint_hash"]
    recs[1]["rle"] = {"size": [48, 64], "counts": "not-rle"}
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    manifest = export_campaign(c / "worklist.json", c)
    joined = " ".join(manifest["integrity"]["problems"])
    assert "missing provenance" in joined and ("does not decode" in joined or "mask" in joined)
    assert manifest["integrity"]["invalid_records_excluded"] == 1 and not manifest["integrity"]["ok"]
    assert manifest["masks"] == 22 + 24 - 1                      # the broken record is not exported


def test_export_without_integrity_and_with_pending(roots, tmp_path):
    c = tmp_path / "campaign"
    save_worklist(c / "worklist.json", build_worklist(roots, "all"))
    manifest = export_campaign(c / "worklist.json", c, check_integrity=False)
    assert manifest["clips"] == {"pending": 4} and manifest["masks"] == 0
    assert manifest["integrity"] == {"checked": False, "clips_checked": 0, "ok": True,
                                     "invalid_records_excluded": 0, "problem_count": 0, "problems": []}
