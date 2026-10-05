"""The forward + backward merge (the pixels rule): the reading of the two candidates, the
handover of shared pixels, the refusals, and the whole-clip walk over the frames that get a
record."""
import numpy as np

from vidstg_masks import merge
from vidstg_masks.worker import build_worklist, plan_unit


def square(y0, y1, x0, x1, shape=(48, 64)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


BOX = dict(xmin=10, ymin=10, xmax=29, ymax=29)          # 20 x 20, inclusive


def test_inside_share_and_iou():
    m = square(10, 30, 10, 30)
    assert merge.inside_share(m, BOX) == 1.0
    assert merge.inside_share(square(10, 30, 20, 40), BOX) == 0.5        # half the pixels left of x=30
    assert merge.inside_share(m, None) is None
    assert merge.iou(m, m) == 1.0 and merge.iou(m, square(10, 30, 20, 40)) == round(200 / 600, 4)
    assert merge.iou(m, np.zeros_like(m)) == 0.0


def test_read_candidates():
    rc = lambda fa, ba, iou_v=None: merge.read_candidates(fa, ba, iou_v, 0.3, 20, 0.1)
    assert rc(0, 0) == ("both_empty", None)
    assert rc(5, 0) == ("forward_speck_only", None) and rc(0, 5) == ("backward_speck_only", None)
    assert rc(5, 7) == ("both_speck", None)
    assert rc(400, 0) == ("forward_only", "forward") and rc(0, 400) == ("backward_only", "backward")
    assert rc(400, 19) == ("backward_speck", "forward") and rc(19, 400) == ("forward_speck", "backward")
    assert rc(400, 39) == ("backward_speck", "forward")           # under a tenth of the other
    assert rc(39, 400) == ("forward_speck", "backward")
    assert rc(400, 40, 0.1) == ("dispute", "dispute")            # a tenth exactly is a real mask
    assert rc(400, 300, 0.3) == ("agree", "either")
    assert rc(400, 300, 0.29) == ("dispute", "dispute")


def test_merge_frame_handover_baby_and_adult():
    """The baby (t0) agrees with itself but its forward mask covers the adult's arm; the
    adult's (t1) forward mask is a speck. The arm goes to the adult, the baby keeps the rest."""
    body = square(0, 32, 0, 32)                 # 1024 px
    arm = square(0, 32, 32, 48)                 # 512 px
    fwd = {0: (body | arm, 0.9), 1: (square(10, 13, 40, 43), 0.95)}    # 9 px speck for the adult
    bwd = {0: (body | arm, 0.8), 1: (arm, 0.85)}
    chosen, prov = merge.merge_frame(fwd, bwd, {0: BOX, 1: None})
    assert prov[0]["rule"] == "agree" and prov[0]["decision"] == "forward" and prov[0]["source"] == "forward"
    assert prov[0]["handover"] == dict(removed_px=512, to=[1], area_before=1536, area_after=1024)
    assert prov[1]["rule"] == "forward_speck" and prov[1]["decision"] == "backward"
    assert chosen[0][0].sum() == 1024 and chosen[0][1] == 0.9           # the forward pass's own confidence
    assert chosen[1][0].sum() == 512 and chosen[1][1] == 0.85
    assert not (chosen[0][0] & chosen[1][0]).any()
    assert prov[1]["forward"]["area"] == 9 and prov[1]["backward"]["inside"] is None
    assert (prov[0]["agree_iou"], prov[0]["speck_floor_px"], prov[0]["speck_ratio"]) == (0.3, 20, 0.1)
    # the forward side may arrive compressed
    chosen2, prov2 = merge.merge_frame({0: (merge.encode(body | arm), 0.9), 1: (merge.encode(fwd[1][0]), 0.95)},
                                       bwd, {0: BOX, 1: None})
    assert chosen2[0][0].sum() == 1024 and prov2[0]["handover"]["removed_px"] == 512


def test_merge_frame_single_object_rules():
    f = square(10, 30, 10, 30)
    assert merge.merge_frame({}, {}, {0: BOX})[1][0]["rule"] == "both_empty"
    chosen, p = merge.merge_frame({0: (f, 0.9)}, {}, {0: BOX})
    assert p[0]["rule"] == "forward_only" and chosen[0][1] == 0.9 and p[0]["backward"] is None
    chosen, p = merge.merge_frame({}, {0: (f, 0.8)}, {0: BOX})
    assert p[0]["rule"] == "backward_only" and chosen[0][1] == 0.8
    # an empty backward mask counts as no mask
    assert merge.merge_frame({0: (f, 0.9)}, {0: (np.zeros_like(f), 0.1)}, {0: BOX})[1][0]["rule"] == "forward_only"
    # a speck is no mask: nothing written, provenance says why
    chosen, p = merge.merge_frame({0: (square(0, 3, 0, 3), 0.9)}, {}, {0: BOX})
    assert chosen == {} and p[0] == dict(p[0], rule="forward_speck_only", decision="none")
    # the masks mostly match -> forward kept
    chosen, p = merge.merge_frame({0: (f, 0.9)}, {0: (square(10, 30, 10, 31), 0.8)}, {0: BOX})
    assert p[0]["rule"] == "agree" and p[0]["source"] == "forward" and chosen[0][1] == 0.9 and p[0]["iou"] > 0.9
    # two real masks that do not overlap -> refused; no box, no tie-break
    far = square(35, 45, 40, 60)
    chosen, p = merge.merge_frame({0: (f, 0.9)}, {0: (far, 0.9)}, {0: BOX})
    assert chosen == {} and p[0]["rule"] == "dispute" and p[0]["decision"] == "refused"
    assert p[0]["refused_reason"] == "disputed_mask" and p[0]["disputed"] is True


def test_merge_frame_conflict_fallback_and_handed_over_speck():
    f = square(10, 30, 10, 30)
    # two strong votes from different passes on the same pixels refuse both
    chosen, p = merge.merge_frame({0: (f, 0.9)}, {1: (square(20, 40, 20, 40), 0.9)}, {0: BOX, 1: BOX})
    assert chosen == {} and p[0]["refused_reason"] == p[1]["refused_reason"] == "passes_conflict"
    # a weak mask trimmed under the floor falls back to its backward mask when that touches nothing
    F = square(0, 6, 0, 10)                                 # 60 px forward
    B0 = square(0, 2, 0, 10); B0[1, 9] = False; B0[6, 0] = True   # 20 px: 19 inside F, 1 outside -> IoU 0.311
    fwd = {0: (F, 0.9), 1: (square(20, 21, 20, 23), 0.9)}  # t1 forward: 3 px speck
    bwd = {0: (B0, 0.8), 1: (F & ~B0, 0.8)}                # t1 backward: the 41 px of F outside B0
    chosen, p = merge.merge_frame(fwd, bwd, {0: None, 1: None})
    assert p[0]["rule"] == "agree" and p[0]["decision"] == "backward"
    assert p[0]["handover"]["area_after"] == 19 and p[0]["handover"]["fallback"] == "backward"
    assert chosen[0][0].sum() == 20 and chosen[0][1] == 0.8 and not (chosen[0][0] & chosen[1][0]).any()
    # ... and is refused when the backward mask touches a written one
    bwd = {0: (B0, 0.8), 1: (F, 0.8)}
    chosen, p = merge.merge_frame(fwd, bwd, {0: None, 1: None})
    assert 0 not in chosen and p[0]["refused_reason"] == "handed_over_speck" and p[0]["handover"]["area_after"] == 0


def test_compress_drops_empty_masks_and_round_trips():
    m = square(0, 5, 0, 5)
    out = merge.compress({3: {0: (m, 0.9), 1: (np.zeros_like(m), 0.2)}, 4: {}})
    assert set(out[3]) == {0} and out[4] == {} and merge.decode(out[3][0][0]).sum() == 25


def test_merge_passes_walks_the_record_frames(roots):
    wl = build_worklist(roots, "train")
    plan, _ = plan_unit(wl["units"][0], roots, 16, "human")
    assert plan["anchors"][2] is None                         # tid 2 is refused: never merged
    f = square(10, 30, 10, 30)
    g = square(35, 45, 40, 60)                                # tid 1 sits elsewhere: no shared pixels
    fwd = merge.compress({fid: {0: (f, 0.9), 1: (g if fid != 5 else np.zeros_like(f), 0.8)} for fid in range(11)})
    bwd = {fid: {0: (f, 0.7), 1: (g, 0.6)} for fid in range(1, 12)}     # the backward pass misses frame 0
    merged, prov, counts = merge.merge_passes(fwd, bwd, plan)
    keys = set(prov)
    assert all(t in (0, 1) for t, _ in keys) and len(keys) == 22    # tid 1 spans frames 2..11
    assert prov[(1, 5)]["rule"] == "backward_only" and merged[5][1][1] == 0.6
    assert prov[(0, 0)]["rule"] == "forward_only" and prov[(0, 11)]["rule"] == "backward_only"
    assert prov[(0, 3)]["rule"] == "agree" and merged[3][0][1] == 0.9
    assert counts == {"agree": 18, "forward_only": 1, "backward_only": 3,
                      "decision:forward": 19, "decision:backward": 3}
