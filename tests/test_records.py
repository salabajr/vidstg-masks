import json

import numpy as np
import pytest

from vidstg_masks.records import (ANCHOR_POLICIES, PROMPT_MODES, PROVENANCE_FIELDS, REASON_CODES,
                                  make_record, read_jsonl, rle_decode, rle_encode, validate_record,
                                  write_jsonl_atomic)

BOX = {"tid": 1, "bbox": {"xmin": 1, "ymin": 2, "xmax": 3, "ymax": 4}, "generated": 1, "tracker": "kcf"}
PAYLOAD = {"anchor_fids": [0, 4], "ref_anchor_fid": 0, "anchor_policy": "human"}


def test_rle_roundtrip():
    rng = np.random.default_rng(1)
    m = rng.random((48, 64)) > 0.6
    rle = rle_encode(m)
    assert rle["size"] == [48, 64] and isinstance(rle["counts"], str)
    assert np.array_equal(rle_decode(rle), m)
    empty = rle_encode(np.zeros((5, 7), bool))
    assert not rle_decode(empty).any() and rle_decode(empty).shape == (5, 7)


def test_make_record_has_exactly_the_provenance_fields(base_prov):
    rec = make_record("v", 1, 7, box=BOX, split="val", prompt_mode="pvs_box_multianchor",
                      prompt_payload=PAYLOAD, base_prov=base_prov,
                      rle=rle_encode(np.ones((2, 2), bool)), mask_confidence=0.9)
    assert set(rec) == set(PROVENANCE_FIELDS)
    assert rec["box_generated"] == 1 and rec["box_tracker"] == "kcf"
    assert rec["model"] == "sam3.1-object-multiplex" and rec["sam_version"] == "3.1"
    validate_record(rec)


def test_refusal_record(base_prov):
    rec = make_record("v", 1, 7, box=BOX, split="val", prompt_mode="pvs_box_multianchor",
                      prompt_payload=PAYLOAD, base_prov=base_prov, reason_code="empty_mask")
    assert rec["rle"] is None and rec["mask_confidence"] is None
    validate_record(rec)
    with pytest.raises(ValueError):
        make_record("v", 1, 7, box=BOX, split="val", prompt_mode="pvs_box_multianchor",
                    prompt_payload=PAYLOAD, base_prov=base_prov)  # neither rle nor reason
    bad = dict(rec, reason_code="made_up")
    with pytest.raises(ValueError, match="reason_code"):
        validate_record(bad)


@pytest.mark.parametrize("field", PROVENANCE_FIELDS)
def test_missing_provenance_field_fails(base_prov, field):
    rec = make_record("v", 1, 7, box=BOX, split="val", prompt_mode="pvs_box_multianchor",
                      prompt_payload=PAYLOAD, base_prov=base_prov,
                      rle=rle_encode(np.ones((2, 2), bool)), mask_confidence=0.5)
    del rec[field]
    with pytest.raises(ValueError, match="missing provenance"):
        validate_record(rec)


def test_validation_catches_bad_values(base_prov):
    good = make_record("v", 1, 7, box=BOX, split="val", prompt_mode="pvs_box_multianchor",
                       prompt_payload=PAYLOAD, base_prov=base_prov,
                       rle=rle_encode(np.ones((2, 2), bool)), mask_confidence=0.5)
    for k, v in [("split", "calib"), ("prompt_mode", "text"), ("box_generated", 2),
                 ("checkpoint_hash", "abc"), ("mask_confidence", 1.5), ("mask_confidence", None),
                 ("prompt_payload", {"anchor_fids": []}), ("model", "sam2")]:
        with pytest.raises(ValueError):
            validate_record(dict(good, **{k: v}))
    with pytest.raises(ValueError):
        validate_record(dict(good, reason_code="empty_mask"))  # mask + reason


def test_schema_file_agrees_with_code():
    from pathlib import Path
    schema = json.loads((Path(__file__).resolve().parents[1] / "schema" / "mask_record.schema.json").read_text())
    assert set(schema["required"]) == set(PROVENANCE_FIELDS)
    assert set(schema["properties"]["reason_code"]["enum"]) == set(REASON_CODES)
    assert set(schema["properties"]["prompt_mode"]["enum"]) == set(PROMPT_MODES)
    payload = schema["properties"]["prompt_payload"]["properties"]
    assert set(payload["anchor_policy"]["enum"]) == set(ANCHOR_POLICIES)
    assert {"hq", "gap", "contained_negatives"} <= set(payload)
    from vidstg_masks import anchors
    assert ANCHOR_POLICIES == anchors.ANCHOR_POLICIES
    assert set(PROMPT_MODES) == set(anchors.PROMPT_MODES.values())


def test_write_jsonl_atomic(tmp_path, base_prov):
    recs = [make_record("v", 1, f, box=BOX, split="val", prompt_mode="pvs_box_multianchor",
                        prompt_payload=PAYLOAD, base_prov=base_prov, reason_code="empty_mask")
            for f in range(3)]
    p = tmp_path / "records" / "v.jsonl"
    assert write_jsonl_atomic(p, recs) == 3
    assert list(read_jsonl(p)) == recs
    assert not list(tmp_path.glob("records/*.part"))


def test_validation_of_gap_and_contained_negatives_payloads(base_prov):
    def mask(payload):
        return make_record("v", 1, 7, box=BOX, split="val", prompt_mode="pvs_box_multianchor",
                           prompt_payload=payload, base_prov=base_prov,
                           rle=rle_encode(np.ones((2, 2), bool)), mask_confidence=0.5)
    gap = {"max_gap": 60, "gap_fill": "human", "n_gap_fills": 0, "gap_fills": {}, "baseline_fids": [0, 4]}
    validate_record(mask({**PAYLOAD, "anchor_policy": "human_gap", "gap": gap}))
    for bad in [{**PAYLOAD, "anchor_policy": "human_gap"},                 # gap missing
                {**PAYLOAD, "anchor_policy": "human_gap", "gap": None},    # prompted, no gap
                {**PAYLOAD, "anchor_policy": "human_gap", "gap": [1]},
                {**PAYLOAD, "anchor_policy": "any"},
                {**PAYLOAD, "contained_negatives": [1, 2]}]:
        with pytest.raises(ValueError):
            validate_record(mask(bad))
    # an object never prompted under human_gap (a refusal) carries gap null
    validate_record(make_record("v", 1, 7, box=BOX, split="val", prompt_mode="pvs_box_multianchor",
                                prompt_payload={"anchor_fids": [], "ref_anchor_fid": None,
                                                "anchor_policy": "human_gap", "gap": None},
                                base_prov=base_prov, reason_code="no_human_keyframe"))
    neg = {"neg_clicks": {"0": [[0.3, 0.3, 2]]}, "co_prompt_fids": [3], "co_prompt_tracker_fids": [3],
           "n_neg_clicks": 1}
    validate_record(mask({**PAYLOAD, "contained_negatives": neg}))
    validate_record(mask({**PAYLOAD, "contained_negatives": None}))
