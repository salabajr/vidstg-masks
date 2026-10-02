import pytest

from conftest import VID_A, VID_B
from vidstg_masks.anchors import (add_contained_negatives, apply_anchor_policy, box_iou, contained,
                                  in_span_fids, plan_clip, prompts_for_plan, unpack_prompt)
from vidstg_masks.datasets import build_vidor_index, load_vidor, load_vidstg, records_by_vid


def test_box_iou():
    a = {"xmin": 0, "ymin": 0, "xmax": 9, "ymax": 9}
    assert box_iou(a, a) == 1.0
    assert box_iou(a, {"xmin": 20, "ymin": 20, "xmax": 29, "ymax": 29}) == 0.0
    assert abs(box_iou(a, {"xmin": 5, "ymin": 0, "xmax": 14, "ymax": 9}) - 50 / 150) < 1e-9


def _plan(roots, vid, **kw):
    recs = records_by_vid(load_vidstg(roots))[vid]
    ann = load_vidor(build_vidor_index(roots)[vid])
    return plan_clip(vid, recs, ann, recs[0]["vidstg_split"], **kw)


def test_plan_clip_object_set_segments_and_spans(roots):
    p = _plan(roots, VID_A)
    assert p["relations"] == [(0, "hold", 1), (0, "watch", 2)]
    assert p["tids"] == [0, 1, 2]
    assert p["seg"] == (0, 11) and p["prop_span"] == (0, 11)
    assert p["spans"] == {0: (0, 11), 1: (2, 11), 2: (0, 11)}
    assert p["cats"] == {0: "adult", 1: "ball/sports_ball", 2: "dog"}
    assert p["split"] == "train" and p["anchor_policy"] == "human"


def test_anchors_are_human_keyframes_only_and_ref_rule_holds(roots):
    p = _plan(roots, VID_A)
    for t in (0, 1):
        for f in p["anchors"][t]["fids"]:
            assert p["boxes"][t][f]["generated"] == 0
    assert p["anchors"][0]["fids"] == [0, 4, 8]
    assert p["anchors"][1]["fids"] == [2, 6, 10]
    # tid 1 overlaps tid 0 at 2 and 10 but not at 6 -> most discriminative keyframe
    assert p["anchors"][1]["ref"] == 6 and p["anchors"][1]["ref_max_iou"] == 0.0
    # tid 0: no other box at fid 0 -> IoU 0; ties broken earliest
    assert p["anchors"][0]["ref"] == 0
    # tid 2 has only tracker boxes -> refused before any GPU work
    assert p["anchors"][2] is None and p["refusals"] == {2: "no_human_keyframe"}


def test_max_anchors_thins_but_keeps_ref(roots):
    p = _plan(roots, VID_B, max_anchors=2)
    a0 = p["anchors"][0]
    assert a0["n_human"] == 4                      # fids 0, 3, 6, 9
    assert a0["ref"] in a0["fids"] and len(a0["fids"]) <= 3
    assert a0["fids"][0] == 0 and a0["fids"][-1] == 9
    p_all = _plan(roots, VID_B, max_anchors=0)
    assert p_all["anchors"][0]["fids"] == [0, 3, 6, 9]


def test_prompts_reference_first_relative_coords(roots):
    p = _plan(roots, VID_A)
    prompts = prompts_for_plan(p)
    assert list(prompts) == [0, 1]                 # refused tid 2 absent
    fids = [f for f, _ in prompts[1]]
    assert fids == [6, 2, 10]                      # ref first, then fid order
    fid, b = prompts[1][0]
    assert b == [40 / 64, 10 / 48, 60 / 64, 30 / 48]
    assert all(0.0 <= v <= 1.0 for f, bb in prompts[0] for v in bb)


def test_in_span_fids_are_boxed_frames_inside_span(roots):
    p = _plan(roots, VID_A)
    assert in_span_fids(p, 1) == list(range(2, 12))
    assert in_span_fids(p, 0) == list(range(12))


def test_unknown_policy_rejected(roots):
    with pytest.raises(ValueError):
        _plan(roots, VID_A, anchor_policy="any")
    p = _plan(roots, VID_A)
    assert apply_anchor_policy(p, "human", None) is p


# ── human_gap: the coverage rule alone, no gate, no video ────────────────────

def test_human_gap_policy_fills_gaps_with_human_keyframes(roots):
    """VID_B tid 0 has human keyframes 0/3/6/9, tid 1 has 0/4/8; max_anchors 2 thins them to
    the first and last, and the gap rule (max gap 4) puts the middle ones back."""
    p = _plan(roots, VID_B, max_anchors=2)
    assert p["anchors"][0]["fids"] == [0, 9] and p["anchors"][1]["fids"] == [0, 8]
    g = apply_anchor_policy(p, "human_gap", None, max_anchors=2, max_gap=4)   # no video needed
    assert g["anchor_policy"] == "human_gap" and g["anchor_mode"] == "human_gap"
    assert p["anchor_policy"] == "human" and p["anchors"][0]["fids"] == [0, 9]   # input untouched
    a0 = g["anchors"][0]
    assert a0["fids"] == [0, 3, 6, 9] and a0["ref"] == 0 and a0["anchor_policy"] == "human_gap"
    assert a0["gap"] == {"max_gap": 4, "gap_fill": "human", "n_gap_fills": 2,
                         "gap_fills": {"3": {"source": "human", "flags": [], "gap": [0, 9]},
                                       "6": {"source": "human", "flags": [], "gap": [3, 9]}},
                         "baseline_fids": [0, 9]}
    assert a0["n_tracker_anchors"] == 0
    assert g["anchors"][1]["fids"] == [0, 4, 8] and g["anchors"][1]["gap"]["n_gap_fills"] == 1
    # no stretch longer than max_gap where a human keyframe could have filled it
    for t in (0, 1):
        lo, hi = g["spans"][t]
        pts = [lo] + g["anchors"][t]["fids"] + [hi]
        for a, b in zip(pts, pts[1:]):
            human_inside = [f for f in range(a + 1, b) if g["boxes"][t][f]["generated"] == 0]
            assert b - a <= 4 or not human_inside, (t, a, b)
    # prompts keep the reference first, then fid order, all human boxes
    assert [f for f, _ in prompts_for_plan(g)[0]] == [0, 3, 6, 9]
    assert all(g["boxes"][0][f]["generated"] == 0 for f in a0["fids"])
    # max_gap 0: the human plan's anchors, recorded as such
    off = apply_anchor_policy(p, "human_gap", None, max_anchors=2, max_gap=0)
    assert off["anchors"][0]["fids"] == [0, 9] and off["anchors"][0]["gap"]["n_gap_fills"] == 0
    # a refused object stays refused
    pa = _plan(roots, VID_A)
    ga = apply_anchor_policy(pa, "human_gap", None)
    assert ga["anchors"][2] is None and ga["refusals"] == {2: "no_human_keyframe"}
    assert ga["anchors"][0]["fids"] == [0, 4, 8] and ga["anchors"][0]["gap"]["n_gap_fills"] == 0


# ── contained negatives ──────────────────────────────────────────────────────

def test_unpack_prompt():
    b = [0.1, 0.2, 0.3, 0.4]
    assert unpack_prompt((3, b)) == (3, b, [], [])
    fid, box, pts, labels = unpack_prompt((3, b, ([0.5, 0.5],), (0,)))
    assert (fid, box, pts, labels) == (3, b, [[0.5, 0.5]], [0]) and isinstance(labels, list)


def test_contained_geometry():
    outer = dict(xmin=10, ymin=10, xmax=69, ymax=69)                          # 60 x 60 = 3600 px
    assert contained(outer, dict(xmin=30, ymin=30, xmax=39, ymax=39))         # inside, 100 px
    assert not contained(outer, dict(xmin=10, ymin=10, xmax=49, ymax=49))     # inside but 1600 px > 25%
    assert not contained(outer, dict(xmin=65, ymin=65, xmax=74, ymax=74))     # only 25 of 100 px inside
    assert not contained(outer, dict(xmin=100, ymin=100, xmax=109, ymax=109))  # outside
    assert contained(outer, dict(xmin=64, ymin=30, xmax=73, ymax=39), min_inside=0.6)   # 60 px inside
    assert not contained(outer, dict(xmin=64, ymin=30, xmax=73, ymax=39))
    assert not contained(dict(xmin=30, ymin=30, xmax=39, ymax=39), outer)     # not symmetric


def container_plan():
    """tid 0 (a person, 60x60 box) and tid 1 (a toy, 10x10 box inside it) boxed on frames 0..10 of
    a 100x100 video. tid 0 is anchored at 0 and 5 (human boxes; 8 is human too, 2 is a tracker
    box); tid 1 at 2 and 8."""
    big = dict(xmin=10, ymin=10, xmax=69, ymax=69)
    small = dict(xmin=30, ymin=30, xmax=39, ymax=39)
    boxes = {0: {f: dict(bbox=dict(big), generated=0 if f in (0, 5, 8) else 1,
                         tracker="none" if f in (0, 5, 8) else "kcf") for f in range(11)},
             1: {f: dict(bbox=dict(small), generated=0 if f in (2, 8) else 1,
                         tracker="none" if f in (2, 8) else "kcf") for f in range(11)}}
    return dict(vid="x", split="val", W=100, H=100, fps=10.0, frame_count=11, video_path_rel="x.mp4",
                relations=[(0, "hold", 1)], tids=[0, 1], cats={0: "adult", 1: "toy"}, seg=(0, 10),
                spans={0: (0, 10), 1: (0, 10)}, boxes=boxes, prop_span=(0, 10),
                anchors={0: dict(ref=0, ref_max_iou=0.0, fids=[0, 5], n_human=3, anchor_policy="human"),
                         1: dict(ref=2, ref_max_iou=0.0, fids=[2, 8], n_human=2, anchor_policy="human")},
                refusals={}, anchor_policy="human", max_anchors=16)


def test_contained_negatives_click_and_co_prompt_the_container():
    plan = container_plan()
    prompts = prompts_for_plan(plan)
    prov = add_contained_negatives(plan, prompts)
    c = [34.5 / 100, 34.5 / 100]                     # centre of the toy box, relative
    big = [0.1, 0.1, 0.69, 0.69]
    small = [0.3, 0.3, 0.39, 0.39]
    # 1. the container gets a negative click at the toy's centre at its own anchors, ref first ...
    assert prompts[0][:2] == [(0, big, [c], [0]), (5, big, [c], [0])]
    # 2. ... and is co-prompted at the toy's anchors with its own box plus the negative
    assert prompts[0][2:] == [(2, big, [c], [0]), (8, big, [c], [0])]
    # the toy contains nothing: its entries stay the plain 2-tuples
    assert prompts[1] == [(2, small), (8, small)]
    assert prov[0] == dict(neg_clicks={"0": [[*c, 1]], "2": [[*c, 1]], "5": [[*c, 1]], "8": [[*c, 1]]},
                           co_prompt_fids=[2, 8], co_prompt_tracker_fids=[2], n_neg_clicks=4)
    assert prov[1] == dict(neg_clicks={}, co_prompt_fids=[], co_prompt_tracker_fids=[], n_neg_clicks=0)
    # a container already anchored at the toy's frame gets the click there, not a second entry
    plan2 = container_plan()
    plan2["anchors"][0]["fids"] = [0, 2]
    prompts2 = prompts_for_plan(plan2)
    prov2 = add_contained_negatives(plan2, prompts2)
    assert [f for f, *_ in prompts2[0]] == [0, 2, 8] and prov2[0]["co_prompt_fids"] == [8]


def test_contained_negatives_leave_prompts_alone_without_containment(roots):
    p = _plan(roots, VID_A)                          # tid 1's box is 65% of tid 0's: not "small"
    prompts = prompts_for_plan(p)
    before = {t: list(v) for t, v in prompts.items()}
    prov = add_contained_negatives(p, prompts)
    assert prompts == before
    assert set(prov) == {0, 1}                       # refused tid 2 has no prompts, so no entry
    assert all(v["n_neg_clicks"] == 0 and v["co_prompt_fids"] == [] for v in prov.values())
