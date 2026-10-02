import json

import pytest

from conftest import VID_A, VID_B, VID_C, VID_D
from vidstg_masks.datasets import (KNOWN_FACTS, Roots, boxes_for_tid, build_vidor_index,
                                   known_facts_check, load_vidor, load_vidstg, records_by_vid,
                                   relation_tids, resolve_video)


def test_roots_from_env(data):
    r = Roots.from_env()
    assert r.vidstg_root == data["vidstg"]
    assert r.vidor_ann_root == data["vidor_ann"]
    assert r.vidor_video_root == data["videos"]
    assert r.vidor_transcoded_root == data["transcoded"]
    assert Roots.from_dict(r.to_dict()) == r


def test_roots_missing_env_is_an_error(monkeypatch):
    for k in Roots.ENV:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(KeyError, match="VIDSTG_ROOT"):
        Roots.from_env()


def test_roots_cli_overrides_env(data, tmp_path):
    class Args:
        vidstg_root = tmp_path / "elsewhere"
        vidor_ann_root = None
        vidor_video_root = None
        vidor_transcoded_root = None
    r = Roots.from_args(Args())
    assert r.vidstg_root == tmp_path / "elsewhere"
    assert r.vidor_ann_root == data["vidor_ann"]


def test_vidstg_loads_all_three_splits(roots):
    recs = load_vidstg(roots)
    splits = {r["vidstg_split"] for r in recs}
    assert splits == {"train", "val", "test"}
    assert {r["vid"] for r in recs if r["vidstg_split"] == "test"} == {VID_C}
    assert {r["vid"] for r in recs if r["vidstg_split"] == "val"} == {VID_B, VID_D}
    assert len(records_by_vid(recs)[VID_A]) == 2


def test_join_finds_every_relation_tid(roots):
    index = build_vidor_index(roots)
    assert set(index) >= {VID_A, VID_B, VID_C, VID_D, "9999999999"}
    for vid, recs in records_by_vid(load_vidstg(roots)).items():
        ann = load_vidor(index[vid])
        for tid in relation_tids(recs):
            assert boxes_for_tid(ann, tid), f"{vid} tid {tid} has no boxes"
    a = load_vidor(index[VID_A])
    assert sorted(boxes_for_tid(a, 1)) == list(range(2, 12))
    assert boxes_for_tid(a, 0)[4]["generated"] == 0 and boxes_for_tid(a, 0)[5]["tracker"] == "kcf"


def test_resolve_video_prefers_transcoded(roots, data):
    ann = data["anns"][VID_A]
    assert resolve_video(roots, ann) == data["videos"] / "0001" / f"{VID_A}.mp4"
    t = data["transcoded"] / "0001" / f"{VID_A}.mp4"
    t.parent.mkdir(parents=True)
    t.write_bytes(b"x")
    assert resolve_video(roots, ann) == t
    assert resolve_video(roots, data["anns"][VID_D]) is None


def test_known_facts_check_reports_counts(roots):
    facts = known_facts_check(load_vidstg(roots), build_vidor_index(roots))
    assert facts["counts"]["vidstg_records"] == 20
    assert facts["counts"]["vidstg_videos"] == 4
    assert facts["counts"]["relation_pairs"] == 3 + 2 + 17 + 2
    assert facts["expected"] == KNOWN_FACTS
    # the fixture patches only the VidOR file count to its own size (conftest)
    assert facts["ok"] is False
    assert facts["matches"] == {"vidstg_records": False, "vidstg_videos": False,
                                "relation_pairs": False, "vidor_annotations": True}
    assert facts["videos_with_vidor_annotation"] == 4


def test_vidor_index_is_depth_tolerant_and_numeric_only(roots, data):
    import json as _json
    index = build_vidor_index(roots)
    assert len(index) == 5 and all(k.isdigit() for k in index)
    # a copy nested one level deeper, or flattened, still indexes; non-numeric json is ignored
    deeper = data["vidor_ann"] / "training" / "extra" / "0009"
    deeper.mkdir(parents=True)
    (deeper / "1000000009.json").write_text(_json.dumps(data["anns"][VID_A]))
    (data["vidor_ann"] / "1000000010.json").write_text(_json.dumps(data["anns"][VID_A]))
    (data["vidor_ann"] / "training" / "0001" / "index.json").write_text("{}")
    index = build_vidor_index(roots)
    assert set(index) == {VID_A, VID_B, VID_C, VID_D, "9999999999", "1000000009", "1000000010"}
