"""The forward + backward merge: every rule, the tie-break, the refusal variant, and the
whole-clip walk over the frames that get a record."""
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


def test_merge_frame_rules():
    f = square(10, 30, 10, 30)
    assert merge.merge_frame(None, None, BOX)[1]["rule"] == "both_empty"
    chosen, p = merge.merge_frame((f, 0.9), None, BOX)
    assert p["rule"] == "forward_only" and chosen[1] == 0.9 and p["backward"] is None
    chosen, p = merge.merge_frame(None, (f, 0.8), BOX)
    assert p["rule"] == "backward_only" and chosen[1] == 0.8
    # an empty backward mask counts as no mask
    assert merge.merge_frame((f, 0.9), (np.zeros_like(f), 0.1), BOX)[1]["rule"] == "forward_only"
    # agree: nearly the same mask -> forward kept
    chosen, p = merge.merge_frame((f, 0.9), (square(10, 30, 10, 31), 0.8), BOX)
    assert p["rule"] == "agree" and p["source"] == "forward" and chosen[1] == 0.9 and p["iou"] > 0.9
    # tiebreak: overlap 0.5, the backward mask is half outside the box -> forward wins on inside share
    b = square(10, 30, 20, 40)
    chosen, p = merge.merge_frame((f, 0.9), (b, 0.95), BOX)
    assert p["rule"] == "tiebreak" and p["source"] == "forward" and p["forward"]["inside"] == 1.0
    assert p["backward"]["inside"] == 0.5 and chosen[1] == 0.9
    # the same frame with the roles swapped -> the backward mask wins
    chosen, p = merge.merge_frame((b, 0.95), (f, 0.9), BOX)
    assert p["rule"] == "tiebreak" and p["source"] == "backward" and chosen[1] == 0.9
    # equal inside shares -> the larger mask
    big = square(10, 30, 10, 30); big[12:28, 12:28] = True
    small = square(12, 28, 12, 28)
    chosen, p = merge.merge_frame((small, 0.5), (big, 0.5), BOX)
    assert p["rule"] == "tiebreak" and p["source"] == "backward" and p["backward"]["area"] > p["forward"]["area"]
    # disputed: no overlap -> chosen by the tie-break, flagged
    far = square(35, 45, 40, 60)
    chosen, p = merge.merge_frame((f, 0.9), (far, 0.9), BOX)
    assert p["rule"] == "disputed" and p["disputed"] is True and p["source"] == "forward" and chosen is not None
    # ... or refused
    chosen, p = merge.merge_frame((f, 0.9), (far, 0.9), BOX, refuse_disputed=True)
    assert chosen is None and p["rule"] == "disputed_refused" and p["source"] is None
    # thresholds travel in the provenance
    assert (p["agree_iou"], p["dispute_iou"]) == (0.7, 0.3)
    # the forward side may arrive compressed
    chosen, p = merge.merge_frame((merge.encode(f), 0.9), (f, 0.8), BOX)
    assert p["rule"] == "agree" and chosen[0].sum() == f.sum()


def test_compress_drops_empty_masks_and_round_trips():
    m = square(0, 5, 0, 5)
    out = merge.compress({3: {0: (m, 0.9), 1: (np.zeros_like(m), 0.2)}, 4: {}})
    assert set(out[3]) == {0} and out[4] == {} and merge.decode(out[3][0][0]).sum() == 25


def test_merge_passes_walks_the_record_frames(roots):
    wl = build_worklist(roots, "train")
    plan, _ = plan_unit(wl["units"][0], roots, 16, "human")
    assert plan["anchors"][2] is None                         # tid 2 is refused: never merged
    f = square(10, 30, 10, 30)
    fwd = merge.compress({fid: {0: (f, 0.9), 1: (f if fid != 5 else np.zeros_like(f), 0.8)} for fid in range(11)})
    bwd = {fid: {0: (f, 0.7), 1: (f, 0.6)} for fid in range(1, 12)}     # the backward pass misses frame 0
    merged, prov, rules = merge.merge_passes(fwd, bwd, plan)
    keys = set(prov)
    assert all(t in (0, 1) for t, _ in keys) and len(keys) == 22    # tid 1 spans frames 2..11
    assert prov[(1, 5)]["rule"] == "backward_only" and merged[5][1][1] == 0.6
    assert prov[(0, 0)]["rule"] == "forward_only" and prov[(0, 11)]["rule"] == "backward_only"
    assert prov[(0, 3)]["rule"] == "agree" and merged[3][0][1] == 0.9
    assert rules == {"agree": 18, "forward_only": 1, "backward_only": 3}
