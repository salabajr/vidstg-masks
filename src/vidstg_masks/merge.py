"""Merge the forward and the backward pass of one clip into one set of masks: the pixels rule.

Both passes prompt SAM with the same boxes; the forward pass propagates from the span start,
the backward pass from the span end (`sam_session.run_session(direction=...)`). Near an
anchor the pass that has just left it holds the fresher memory, so each pass fails on
different frames. The merge keeps one mask per (object, frame), uses no box, takes no speck,
never paints a pixel twice, and writes what it did into every record (`prompt_payload.merge`).

Reading the two candidates of one object on one frame (`read_candidates`):

  a mask under speck_floor pixels is no mask; with two real masks, one under speck_ratio of
  the other is no mask
  no real mask in either pass          -> nothing written          both_empty, both_speck,
                                                                   forward_speck_only, backward_speck_only
  forward only                         -> forward, a strong vote   forward_only, backward_speck
  backward only                        -> backward, a strong vote  backward_only, forward_speck
  both, IoU >= agree_iou               -> forward preferred, either allowed (a weak vote)   agree
  both, IoU < agree_iou                -> a real dispute: refused, disputed_mask            dispute

Then per frame, across the objects (`decide_pixels`): each object takes its candidate; every
pixel two written masks share goes to the object whose pass was the only candidate (the
strong vote), and the object that merely preferred forward loses those pixels (its own
backward mask, from the winner's pass, also leaves them out: two of the three masks on the
frame say the pixels are not its). Two strong votes from different passes on the same pixels
refuse both objects (passes_conflict). A trimmed mask that falls under speck_floor takes the
object's backward mask when that touches nothing written, else it is refused
(handed_over_speck). Inside one pass SAM gives a pixel to one object only, so after the
handover no two written masks share a pixel.

Measured on 20 VidSTG-val clips (69,729 object-frames; the research repo's
reports/merge_rules_v2.md): 217 refusals (215 disputes, 2 conflicts), 222 object-frames with
nothing real in either pass, 176 masks from the backward pass, 180 masks trimmed, 0 shared
pixels; the earlier per-object rule with a box tie-break left 282 overlapping pairs.

Optional tie-break for disputes (`dispute_rule`; the default `refuse` keeps every dispute refused).
`forward_score`: the forward mask is written when the forward pass's presence score (SAM's per-frame
object score through a sigmoid, recorded as mask_confidence) is at least `dispute_score`.
`higher_score`: the pass with the higher presence score is written when that score is at least
`dispute_score` (equal scores: forward). Below the threshold the dispute stays refused. No box is
read. `dispute_winner` says how the written mask meets the frame's other objects in decide_pixels:
`weak` (default), a vote a strong partner trims, a weak backward vote also yielding to a weak forward
one (forward primary), and what is left under the floor is refused (handed_over_speck), never swapped
for the other disputed mask; `strong`, the winner takes the pixels an agreed neighbour shares with it
(the neighbour is trimmed) and conflicts with another strong vote (passes_conflict).

Measured on the 20 clips against Nathan's 16 firm frame labels (research branch, research/REPORT.md
sections 11 to 14): higher_score at 0.907 with the strong winner agreed on 14, forward_score on 9,
refusing every dispute on 1. The strong winner is what keeps a backward win whole where an agreed
forward neighbour had taken its pixels. 10 of the 16 decisions rest on a score margin under 0.005 and
the threshold on one labeled hidden object (scores 0.81 / 0.86 against 0.956 and up where a mask was
right), so a run's overlays are worth a look before a new set of clips is trusted.

The forward pass is kept compressed (COCO RLE) while the backward pass runs, so a clip costs
the memory of one pass plus its RLE strings. No torch here: the merge is numpy on the CPU.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from pycocotools import mask as mask_util

READINGS = ("both_empty", "forward_speck_only", "backward_speck_only", "both_speck", "forward_only",
            "backward_speck", "backward_only", "forward_speck", "agree", "dispute")
DECISIONS = ("forward", "backward", "refused", "none")
REFUSALS = ("disputed_mask", "passes_conflict", "handed_over_speck")
DEFAULT_AGREE_IOU = 0.3
DEFAULT_SPECK_FLOOR = 20       # pixels
DEFAULT_SPECK_RATIO = 0.1
DISPUTE_RULES = ("refuse", "forward_score", "higher_score")
DISPUTE_WINNERS = ("weak", "strong")
DEFAULT_DISPUTE_RULE = "refuse"     # every dispute is refused
DEFAULT_DISPUTE_SCORE = None        # threshold of the two score rules
DEFAULT_DISPUTE_WINNER = "weak"


def compress(per_frame: dict) -> dict:
    """{fid: {obj: (mask, conf)}} -> {fid: {obj: (rle, conf)}}; empty masks are dropped."""
    out = {}
    for fid, objs in per_frame.items():
        kept = {}
        for obj, (m, conf) in objs.items():
            if m is not None and m.any():
                kept[obj] = (encode(m), conf)
        out[fid] = kept
    return out


def encode(m: np.ndarray) -> dict:
    rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
    return dict(size=[int(x) for x in rle["size"]], counts=rle["counts"].decode())


def decode(rle: dict) -> np.ndarray:
    counts = rle["counts"].encode() if isinstance(rle["counts"], str) else rle["counts"]
    return mask_util.decode(dict(size=rle["size"], counts=counts)).astype(bool)


def inside_share(m: np.ndarray, bbox: dict | None) -> float | None:
    """Share of the mask's pixels inside the (inclusive) VidOR box; None without a box.
    Recorded for the reader of the provenance; the merge does not use it."""
    if bbox is None:
        return None
    H, W = m.shape
    x0, y0 = max(0, int(bbox["xmin"])), max(0, int(bbox["ymin"]))
    x1, y1 = min(W, int(bbox["xmax"]) + 1), min(H, int(bbox["ymax"]) + 1)
    total = int(m.sum())
    if total == 0 or x1 <= x0 or y1 <= y0:
        return 0.0
    return round(float(m[y0:y1, x0:x1].sum()) / total, 4)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return round(float((a & b).sum()) / union, 4) if union else 0.0


def _describe(m: np.ndarray, conf, bbox) -> dict:
    return dict(area=int(m.sum()), inside=inside_share(m, bbox),
                confidence=None if conf is None else float(conf))


def read_candidates(fa: int, ba: int, iou_v, agree: float, floor: int, ratio: float) -> tuple[str, str | None]:
    """The speck rule, then the per-object reading. Returns (reading, pref): pref 'forward' or
    'backward' = that pass is the only real candidate (a strong vote); 'either' = the masks
    mostly match, forward preferred (a weak vote); 'dispute'; None = nothing real to write."""
    if fa == 0 and ba == 0:
        return "both_empty", None
    f_ok, b_ok = fa >= floor, ba >= floor
    if not f_ok and not b_ok:
        return ("both_speck" if fa and ba else "forward_speck_only" if fa else "backward_speck_only"), None
    if f_ok and b_ok:
        if fa < ratio * ba:
            return "forward_speck", "backward"
        if ba < ratio * fa:
            return "backward_speck", "forward"
        return ("agree", "either") if iou_v >= agree else ("dispute", "dispute")
    if f_ok:
        return ("forward_only" if ba == 0 else "backward_speck"), "forward"
    return ("backward_only" if fa == 0 else "forward_speck"), "backward"


def decide_pixels(live: list, pref: dict, cand: dict, floor: int, tiebreak=frozenset()) -> dict:
    """tid -> (decision, refusal reason or None, mask or None, handover dict or None).
    `cand[tid][pass]` is {'mask': bool array, 'area': int, ...} or absent. `pref[tid]`: 'forward' /
    'backward' = that pass was the only candidate (a strong vote); 'either' = the forward mask as a
    weak vote; 'weak_backward' = the backward mask as a weak vote (a dispute tie-break); 'dispute' =
    refused. `tiebreak` names the objects whose mask came from a tie-break: trimmed under the floor
    they are refused, never swapped for the other disputed mask."""
    out, chosen, strong = {}, {}, {}
    for t in live:
        if pref[t] == "dispute":
            out[t] = ("refused", "disputed_mask", None, None)
            continue
        src = "backward" if pref[t] in ("backward", "weak_backward") else "forward"
        chosen[t] = (src, cand[t][src]["mask"].copy())
        strong[t] = pref[t] in ("forward", "backward")
    tids = sorted(chosen)
    handed = defaultdict(lambda: dict(removed_px=0, to=[]))
    conflict = set()
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            (sa, ma), (sb, mb) = chosen[a], chosen[b]
            inter = ma & mb
            n = int(inter.sum())
            if n == 0:
                continue
            if strong[a] and strong[b]:
                conflict.update((a, b))
                continue
            if strong[a] != strong[b]:
                weak, win = (a, b) if strong[b] else (b, a)
            elif sa != sb:                      # two weak votes from different passes: forward keeps the pixels
                weak, win = (a, b) if sb == "forward" else (b, a)
            else:
                continue                        # two weak votes of one pass never share a pixel
            chosen[weak][1][inter] = False
            handed[weak]["removed_px"] += n
            handed[weak]["to"].append(win)
    for t in tids:
        if t in conflict:
            out[t] = ("refused", "passes_conflict", None, None)
            continue
        src, m = chosen[t]
        h = handed.get(t)
        if h is not None:
            h = dict(h, area_before=cand[t][src]["area"], area_after=int(m.sum()))
            if h["area_after"] < floor:
                if t not in tiebreak:
                    mb = cand[t].get("backward")
                    others = [chosen[o][1] for o in tids if o != t and o not in conflict]
                    if mb is not None and not any((mb["mask"] & o).any() for o in others):
                        out[t] = ("backward", None, mb["mask"], dict(h, fallback="backward"))
                        continue
                out[t] = ("refused", "handed_over_speck", None, h)
                continue
        out[t] = (src, None, m, h)
    return out


def check_dispute_settings(rule, score, winner) -> None:
    """The tie-break settings as a set: a score rule needs its threshold, `refuse` takes none."""
    if rule not in DISPUTE_RULES:
        raise ValueError(f"dispute_rule must be one of {DISPUTE_RULES}, got {rule!r}")
    if winner not in DISPUTE_WINNERS:
        raise ValueError(f"dispute_winner must be one of {DISPUTE_WINNERS}, got {winner!r}")
    if rule == "refuse" and score is not None:
        raise ValueError("dispute_score is the threshold of forward_score or higher_score; with dispute_rule "
                         "'refuse' it has no meaning")
    if rule != "refuse" and score is None:
        raise ValueError(f"dispute_rule {rule!r} needs a dispute_score threshold")


def tiebreak_pick(rule: str, forward_score, backward_score, threshold: float) -> str | None:
    """Which pass a dispute tie-break writes: 'forward', 'backward', or None (stays refused).
    forward_score: the forward mask when its score reaches the threshold. higher_score: the pass with
    the higher score when that score reaches the threshold; equal scores go forward. A missing score
    counts as the lowest."""
    cf = -1.0 if forward_score is None else float(forward_score)
    cb = -1.0 if backward_score is None else float(backward_score)
    if rule == "forward_score":
        return "forward" if cf >= threshold else None
    if max(cf, cb) < threshold:
        return None
    return "forward" if cf >= cb else "backward"


def merge_frame(fwd_objs: dict, bwd_objs: dict, boxes: dict, agree_iou: float = DEFAULT_AGREE_IOU,
                speck_floor: int = DEFAULT_SPECK_FLOOR, speck_ratio: float = DEFAULT_SPECK_RATIO,
                dispute_score: float | None = DEFAULT_DISPUTE_SCORE, dispute_rule: str = DEFAULT_DISPUTE_RULE,
                dispute_winner: str = DEFAULT_DISPUTE_WINNER):
    """One frame, every object that gets a record on it. `fwd_objs` / `bwd_objs` map tid ->
    (mask-or-rle, confidence), absent or None where that pass has no mask; `boxes` maps tid ->
    VidOR bbox or None (provenance only). `dispute_rule`, `dispute_score` and `dispute_winner` are
    the tie-break of the module docstring (`refuse` = off). Returns ({tid: (mask, confidence)} for
    the written masks, {tid: provenance dict} for every tid in `boxes`)."""
    check_dispute_settings(dispute_rule, dispute_score, dispute_winner)
    cand, pref, prov = {}, {}, {}
    taken = set()
    for t in sorted(boxes):
        c = {}
        for name, objs in (("forward", fwd_objs), ("backward", bwd_objs)):
            got = objs.get(t)
            if got is None:
                continue
            m = decode(got[0]) if isinstance(got[0], dict) else got[0]
            if m is not None and m.any():
                c[name] = dict(mask=m, conf=got[1], **_describe(m, got[1], boxes[t]))
        cand[t] = c
        fa, ba = c.get("forward", {}).get("area", 0), c.get("backward", {}).get("area", 0)
        iou_v = iou(c["forward"]["mask"], c["backward"]["mask"]) if fa and ba else None
        reading, p = read_candidates(fa, ba, iou_v, agree_iou, speck_floor, speck_ratio)
        tiebreak = None
        if reading == "dispute" and dispute_rule != "refuse":
            cf, cb = c["forward"]["conf"], c["backward"]["conf"]
            pick = tiebreak_pick(dispute_rule, cf, cb, dispute_score)
            tiebreak = dict(rule=dispute_rule, threshold=dispute_score, forward_score=cf, backward_score=cb,
                            pick=pick, taken=pick is not None, winner=dispute_winner)
            if pick is not None:
                # strong: the winner's pass counts as the only candidate in the handover; weak: a vote any
                # strong partner trims, and a weak backward one also yields to a partner's weak forward one
                p = pick if dispute_winner == "strong" else ("either" if pick == "forward" else "weak_backward")
                taken.add(t)
        pref[t] = p
        prov[t] = dict(rule=reading, decision="none", source=None, iou=iou_v, disputed=reading == "dispute",
                       forward=None if "forward" not in c else {k: c["forward"][k] for k in ("area", "inside", "confidence")},
                       backward=None if "backward" not in c else {k: c["backward"][k] for k in ("area", "inside", "confidence")},
                       agree_iou=agree_iou, speck_floor_px=speck_floor, speck_ratio=speck_ratio,
                       dispute_rule=dispute_rule, dispute_score=dispute_score, dispute_winner=dispute_winner,
                       tiebreak=tiebreak)
    live = [t for t in sorted(boxes) if pref[t] is not None]
    chosen = {}
    for t, (dec, reason, m, hand) in decide_pixels(live, pref, cand, speck_floor, taken).items():
        if hand is not None:
            prov[t]["handover"] = hand
        if dec == "refused":
            prov[t].update(decision="refused", refused_reason=reason)
        else:
            prov[t].update(decision=dec, source=dec)
            chosen[t] = (m, cand[t][dec]["conf"])
    return chosen, prov


def merge_passes(fwd: dict, bwd: dict, plan: dict, agree_iou: float = DEFAULT_AGREE_IOU,
                 speck_floor: int = DEFAULT_SPECK_FLOOR, speck_ratio: float = DEFAULT_SPECK_RATIO,
                 dispute_score: float | None = DEFAULT_DISPUTE_SCORE, dispute_rule: str = DEFAULT_DISPUTE_RULE,
                 dispute_winner: str = DEFAULT_DISPUTE_WINNER):
    """Whole clip. `fwd` is the compressed forward pass ({fid: {tid: (rle, conf)}}), `bwd`
    the backward pass ({fid: {tid: (mask, conf)}}). Walks every (tid, fid) with a VidOR box
    inside the tid span (the frames that get a record), frame by frame so the objects of a
    frame are decided together. Returns
    (per_frame {fid: {tid: (mask, conf)}} with the written masks only,
     prov {(tid, fid): merge provenance},
     Counter of readings, decisions ('decision:<name>') and refusal reasons)."""
    from .anchors import in_span_fids

    frames: dict[int, dict] = defaultdict(dict)
    for t in plan["tids"]:
        if plan["anchors"][t] is None:
            continue
        for fid in in_span_fids(plan, t):
            frames[fid][t] = plan["boxes"][t][fid]["bbox"]
    merged: dict[int, dict] = {}
    prov: dict[tuple[int, int], dict] = {}
    counts: Counter = Counter()
    for fid in sorted(frames):
        boxes = frames[fid]
        chosen, p = merge_frame(fwd.get(fid, {}), bwd.get(fid, {}), boxes, agree_iou, speck_floor, speck_ratio,
                                dispute_score, dispute_rule, dispute_winner)
        for t, pt in p.items():
            prov[(t, fid)] = pt
            counts[pt["rule"]] += 1
            counts["decision:" + pt["decision"]] += 1
            if (pt["tiebreak"] or {}).get("taken"):
                counts["tiebreak:" + pt["decision"]] += 1
            if pt["decision"] == "refused":
                counts[pt["refused_reason"]] += 1
        if chosen:
            merged[fid] = chosen
    return merged, prov, counts
