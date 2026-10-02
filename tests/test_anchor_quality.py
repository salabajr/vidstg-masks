"""hq anchor policy on the synthetic clip: gate output shape, provenance, fallback."""

import numpy as np

from conftest import VID_A
from vidstg_masks import anchor_quality as aq
from vidstg_masks.anchors import apply_anchor_policy, box_iou, prompts_for_plan
from vidstg_masks.worker import build_worklist, clip_records, plan_unit


def test_geometry_matches_anchors_module():
    a, b = dict(xmin=0, ymin=0, xmax=9, ymax=9), dict(xmin=5, ymin=5, xmax=14, ymax=14)
    assert aq.box_iou(a, b) == box_iou(a, b) and aq.covered_fraction(a, b) == 0.25
    assert aq.edge_touches(dict(xmin=0, ymin=0, xmax=63, ymax=47), 64, 48) == 4
    assert aq.edge_touches(dict(xmin=10, ymin=10, xmax=30, ymax=30), 64, 48) == 0


def test_hq_policy_gates_human_keyframes_and_records_provenance(roots, base_prov):
    u = build_worklist(roots, "train")["units"][0]
    human, video = plan_unit(u, roots, 16, "human")
    hq = apply_anchor_policy(human, "hq", video, 16)
    assert hq["anchor_policy"] == "hq" and hq["anchor_mode"] == "hq"
    assert human["anchor_policy"] == "human"                         # input plan untouched
    for t in (0, 1):
        a = hq["anchors"][t]
        assert set(a["fids"]) <= set(human["anchors"][t]["fids"])   # subset of the human keyframes
        assert a["ref"] in a["fids"]
        h = a["hq"]
        assert h["n_candidates"] == len(human["anchors"][t]["fids"])
        assert h["baseline_fids"] == human["anchors"][t]["fids"]
        assert set(h) >= {"n_clean", "n_kept", "fallback", "flags_ignored", "dropped", "thresholds"}
    assert hq["anchors"][2] is None and hq["refusals"][2] == "no_human_keyframe"
    # tid 1's fid 6 box is the only non-overlapping keyframe: the gate keeps it as reference
    assert hq["anchors"][1]["ref"] == 6
    prompts = prompts_for_plan(hq)
    assert list(prompts) == [0, 1]
    m = np.ones((48, 64), bool)
    per_frame = {f: {t: (m, 1.0) for t in (0, 1)} for f in range(12)}
    recs, _ = clip_records(hq, per_frame, base_prov)
    r = next(r for r in recs if r["tid"] == 1 and r["rle"] is not None)
    assert r["prompt_mode"] == "pvs_box_hqanchor"
    assert r["prompt_payload"]["anchor_policy"] == "hq" and r["prompt_payload"]["hq"]["n_kept"] == len(hq["anchors"][1]["fids"])
    from vidstg_masks.records import validate_record
    for rec in recs:
        validate_record(rec)


def test_hq_fallback_refuse_gives_no_hq_anchor(roots, base_prov):
    u = build_worklist(roots, "train")["units"][0]
    human, video = plan_unit(u, roots, 16, "human")
    # thresholds nothing can pass: every keyframe is "small" (rel_size < 2 is impossible to beat)
    thr = aq.Thresholds(size_min_rel=2.0)
    metrics = aq.keyframe_metrics(human, thr, None)
    assert all(rows[f]["flags"] for rows in metrics.values() for f in rows)
    refused = aq.apply_hq_policy(human, metrics, thr, 16, "refuse")
    refused["anchor_policy"] = "hq"
    assert refused["anchors"][0] is None and refused["refusals"][0] == "no_hq_anchor"
    recs, refusals = clip_records(refused, None, base_prov)
    assert refusals == {"no_hq_anchor": 22, "no_human_keyframe": 12}
    fallback = aq.apply_hq_policy(human, metrics, thr, 16, "least-flagged")
    assert fallback["anchors"][0]["hq"]["fallback"] is True and "small" in fallback["anchors"][0]["hq"]["flags_ignored"]


def test_blur_scores_need_the_annotated_frame_count(roots, data):
    u = build_worklist(roots, "train")["units"][0]
    human, video = plan_unit(u, roots, 16, "human")
    kf = aq.human_keyframes(human)
    scores = aq.blur_scores(video, human, kf, human["frame_count"])
    assert set(scores) == {(t, f) for t, fs in kf.items() for f in fs}
    human["boxes"][0][20] = dict(human["boxes"][0][8])              # a keyframe past the end of the video
    human["spans"][0] = (0, 20)
    import pytest
    with pytest.raises(RuntimeError, match="do not rescale"):
        aq.blur_scores(video, human, aq.human_keyframes(human), 12)


def _plan_with_keyframes(human, span=(0, 99)):
    """One object boxed on every frame of `span`; `human` are its human keyframes."""
    boxes = {f: {"bbox": {"xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50},
                 "generated": 0 if f in human else 1} for f in range(span[0], span[1] + 1)}
    return dict(vid="x", tids=[0], spans={0: list(span)}, boxes={0: boxes},
                anchors={0: dict(ref=human[0], fids=list(human))}, refusals={})


def test_gap_rule_readmits_the_least_flagged_keyframe_in_the_largest_gap():
    from vidstg_masks.anchor_quality import Thresholds, apply_hq_policy
    plan = _plan_with_keyframes([0, 30, 60, 90])
    metrics = {0: {0: {"flags": []}, 30: {"flags": ["blur"]}, 60: {"flags": ["blur", "edge"]},
                   90: {"flags": []}}}
    out = apply_hq_policy(plan, metrics, Thresholds(max_gap=40), max_anchors=16)
    a = out["anchors"][0]
    assert a["fids"] == [0, 30, 60, 90]                       # both dropped keyframes came back
    fills = a["hq"]["gap_fills"]
    assert fills["30"]["source"] == "human" and fills["30"]["gap"] == [0, 90]   # fewest flags first
    assert fills["60"]["gap"] == [30, 90]
    assert a["hq"]["n_gap_fills"] == 2 and a["n_tracker_anchors"] == 0
    assert "30" not in a["hq"]["dropped"] and a["hq"]["thresholds"]["max_gap"] == 40
    # rule off: only the clean keyframes stay
    off = apply_hq_policy(plan, metrics, Thresholds(max_gap=0), max_anchors=16)
    assert off["anchors"][0]["fids"] == [0, 90] and off["anchors"][0]["hq"]["n_gap_fills"] == 0


def test_gap_rule_covers_the_span_edges_and_falls_back_to_tracker_boxes():
    from vidstg_masks.anchor_quality import Thresholds, apply_hq_policy
    plan = _plan_with_keyframes([50], span=(0, 99))            # one keyframe in the middle
    metrics = {0: {50: {"flags": []}}}
    human_only = apply_hq_policy(plan, metrics, Thresholds(max_gap=20, gap_fill="human"))
    assert human_only["anchors"][0]["fids"] == [50]            # nothing human to fill with
    any_box = apply_hq_policy(plan, metrics, Thresholds(max_gap=20, gap_fill="any"))
    fids = any_box["anchors"][0]["fids"]
    pts = [0] + fids + [99]
    assert max(b - a for a, b in zip(pts, pts[1:])) <= 20      # edges covered too
    assert all(v["source"] == "tracker" for v in any_box["anchors"][0]["hq"]["gap_fills"].values())
    assert any_box["anchors"][0]["n_tracker_anchors"] == len(fids) - 1


def test_span_edges_are_always_anchored():
    from vidstg_masks.anchor_quality import Thresholds, apply_hq_policy
    plan = _plan_with_keyframes([0, 50, 99])
    metrics = {0: {0: {"flags": ["edge"]}, 50: {"flags": []}, 99: {"flags": ["edge", "small"]}}}
    out = apply_hq_policy(plan, metrics, Thresholds(max_gap=0, keep_span_edges=True))
    a = out["anchors"][0]
    assert a["fids"] == [0, 50, 99] and a["hq"]["edge_keep"] == [0, 99]
    assert "0" not in a["hq"]["dropped"] and "99" not in a["hq"]["dropped"]
    off = apply_hq_policy(plan, metrics, Thresholds(max_gap=0, keep_span_edges=False))
    assert off["anchors"][0]["fids"] == [50] and off["anchors"][0]["hq"]["edge_keep"] == []


def test_human_gap_plan_is_the_gap_rule_without_the_gate():
    from vidstg_masks.anchor_quality import human_gap_plan
    plan = _plan_with_keyframes([0, 30, 60, 90])
    plan["anchors"][0]["fids"] = [0, 90]                     # as if thinned to 2
    out = human_gap_plan(plan, 40)
    a = out["anchors"][0]
    assert out["anchor_mode"] == "human_gap" and a["anchor_mode"] == "human_gap"
    assert a["fids"] == [0, 30, 60, 90] and a["ref"] == 0 and a["n_tracker_anchors"] == 0
    assert a["gap"] == dict(max_gap=40, gap_fill="human", n_gap_fills=2, baseline_fids=[0, 90],
                            gap_fills={"30": dict(source="human", flags=[], gap=[0, 90]),
                                       "60": dict(source="human", flags=[], gap=[30, 90])})
    assert plan["anchors"][0]["fids"] == [0, 90] and "gap" not in plan["anchors"][0]   # input untouched
    assert human_gap_plan(plan, 0)["anchors"][0]["fids"] == [0, 90]
    # "any": tracker boxes fill where no human keyframe lies in the gap
    lone = _plan_with_keyframes([50])
    assert human_gap_plan(lone, 20)["anchors"][0]["fids"] == [50]
    any_box = human_gap_plan(lone, 20, "any")["anchors"][0]
    pts = [0] + any_box["fids"] + [99]
    assert max(b - a for a, b in zip(pts, pts[1:])) <= 20
    assert any_box["n_tracker_anchors"] == len(any_box["fids"]) - 1 == any_box["gap"]["n_gap_fills"]


def test_apply_anchor_policy_hands_the_gap_settings_to_the_hq_gate(roots):
    u = build_worklist(roots, "train")["units"][0]
    human, video = plan_unit(u, roots, 16, "human")
    hq = apply_anchor_policy(human, "hq", video, 16, max_gap=7, gap_fill="any", keep_span_edges=True)
    thr = aq.Thresholds(max_gap=7, gap_fill="any", keep_span_edges=True).as_dict()
    assert hq["hq_thresholds"] == thr and hq["anchors"][0]["hq"]["thresholds"] == thr
    default = apply_anchor_policy(human, "hq", video, 16)
    assert default["hq_thresholds"] == aq.Thresholds().as_dict()
