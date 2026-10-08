"""Quality gate over VidOR human keyframes: which keyframes may prompt SAM.

The idea (2026-09-25): the point-to-mask repair redraws a mask from a box
and ten clicks at a frame SAM has never tracked, and that redraw is worse than
the tracked mask it replaces (research review of the SAM API). Instead of repairing,
prompt only where a box will give a clean mask - the object fully in frame,
not overlapped by another relation object, not moving fast, not blurred, not
shrunk by leaving the frame - and let SAM's memory carry that mask to the
frames in between. On the SAM 3.1 multiplex build a prompted frame is always a
from-scratch box segmentation without memory (sam3_tracking_predictor.py:255),
and every un-prompted frame is predicted from memory, so "propagate the good
mask" is exactly "do not prompt on the bad keyframes". No new SAM API, one
propagation, fits a 24 GB card.

This module is CPU only (numpy + cv2, no torch) and self-contained so the
portable pipeline can copy it. `edge_touches` and `motion_at` mirror
the research code; `box_iou` mirrors its box_iou
(pinned equal by tests/test_anchor_quality.py).

Flags per human keyframe (any flag = not a clean anchor):
  edge     the object is being cut by the frame: the box touches more than
           edge_max_touches borders, or touches any border while its area
           is below edge_size_rel x the object's median (leaving/entering).
           Touching one border at full size is NOT a flag - half of all
           VidOR human keyframes do (feet cut off), and prompting there is
           what the baseline always did.
  overlap  another relation object's box covers more than overlap_max of
           this box, or more than overlap_rel_delta above the object's own
           median coverage (a baby held in arms is always ~80% covered; its
           cleanest frames are the least covered ones)
  motion   box centre moves > motion_max box-diagonals per frame (motion blur)
  small    box area < size_min_rel x the object's median human-keyframe area
           (object entering/leaving: the 8295398331 t5 wrist-at-the-edge case)
  blur     Laplacian variance of the grey crop inside the box is below
           blur_abs_min, or below blur_rel_min x the object's own median

Thresholds were set on 2026-09-25 from the metric distributions over 16
VidSTG-val calibration clips (never the test split), chosen
so that a typical object keeps about half of its keyframes and 97 of 104
objects keep at least one clean keyframe. The first attempt (edge on any
border contact, overlap > 0.25, motion > 0.02, blur ratio < 0.5) left 0
clean keyframes for two thirds of the objects: the flags described VidOR
in general, not the bad frames. [INFERRED] these cut-offs separate usable
from unusable prompt frames; the 20-clip video review is the test.

Fallback when an object has no clean keyframe (2026-09-25):
  least-flagged  keep the keyframes with the fewest flags (an object whose box
                 always touches the border keeps its anchors); recorded as
                 hq.fallback = true with the flags that were ignored
  refuse         no anchors -> the object is refused with reason no_hq_anchor
"""

from __future__ import annotations

import math
import statistics as st
from dataclasses import asdict, dataclass

FLAGS = ("edge", "overlap", "motion", "small", "blur")
FALLBACKS = ("least-flagged", "refuse")


@dataclass(frozen=True)
class Thresholds:
    edge_tol_px: int = 2
    edge_max_touches: int = 2      # > this many borders touched -> edge
    edge_size_rel: float = 0.8     # any border touched AND area below this x median -> edge
    overlap_max: float = 0.6       # covered fraction above this -> overlap
    overlap_rel_delta: float = 0.2  # or above the object's median + this -> overlap
    motion_max: float = 0.05
    size_min_rel: float = 0.6
    blur_rel_min: float = 0.4
    blur_abs_min: float = 20.0
    # Coverage rule applied after the gate (see _fill_gaps): no stretch of an object's
    # span longer than max_gap frames without an anchor; 0 disables it. gap_fill "human"
    # re-admits the least-flagged human keyframe in the gap, "any" also allows a tracker
    # box where no human keyframe lies in the gap.
    max_gap: int = 60
    gap_fill: str = "human"
    # Always anchor at the object's first and last human keyframe in its span (what
    # the baseline's thinning does), whatever the gate says about them: SAM tracks
    # forward, so frames before the first anchor must attend to a future memory and
    # are the first to be declared absent; after the last anchor the track drifts.
    keep_span_edges: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


# -- geometry (inclusive pixel coords, as everywhere in this repo) -----------

def box_area(b: dict) -> float:
    return float((b["xmax"] - b["xmin"] + 1) * (b["ymax"] - b["ymin"] + 1))


def box_iou(a: dict, b: dict) -> float:
    iw = min(a["xmax"], b["xmax"]) - max(a["xmin"], b["xmin"]) + 1
    ih = min(a["ymax"], b["ymax"]) - max(a["ymin"], b["ymin"]) + 1
    inter = max(0, iw) * max(0, ih)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union else 0.0


def covered_fraction(a: dict, b: dict) -> float:
    """Share of box a covered by box b (occlusion proxy; not symmetric)."""
    iw = min(a["xmax"], b["xmax"]) - max(a["xmin"], b["xmin"]) + 1
    ih = min(a["ymax"], b["ymax"]) - max(a["ymin"], b["ymin"]) + 1
    return max(0, iw) * max(0, ih) / box_area(a)


def edge_touches(b: dict, W: int, H: int, tol: int = 2) -> int:
    """How many frame borders the box sits on (research code)."""
    return (int(b["xmin"] <= tol) + int(b["ymin"] <= tol)
            + int(b["xmax"] >= W - 1 - tol) + int(b["ymax"] >= H - 1 - tol))


def _centre(b: dict) -> tuple[float, float]:
    return ((b["xmin"] + b["xmax"]) / 2, (b["ymin"] + b["ymax"]) / 2)


def _diag(b: dict) -> float:
    return math.hypot(b["xmax"] - b["xmin"] + 1, b["ymax"] - b["ymin"] + 1)


def motion_at(boxes: dict, fids: list[int], f: int) -> float | None:
    """Centre displacement per frame to the neighbouring annotated frames, in
    box diagonals (research code). None without a neighbour."""
    i = fids.index(f)
    speeds = []
    for j in (i - 1, i + 1):
        if 0 <= j < len(fids):
            g = fids[j]
            cx, cy = _centre(boxes[f]["bbox"])
            gx, gy = _centre(boxes[g]["bbox"])
            d = _diag(boxes[f]["bbox"]) or 1.0
            speeds.append(math.hypot(cx - gx, cy - gy) / (abs(f - g) * d))
    return st.mean(speeds) if speeds else None


# -- keyframe inventory -------------------------------------------------------

def human_keyframes(plan: dict) -> dict[int, list[int]]:
    """{tid: sorted human keyframes (generated == 0) inside the tid span}.
    Same definition as plan_clip's `human` list."""
    out = {}
    for t in plan["tids"]:
        lo, hi = plan["spans"][t]
        out[t] = sorted(f for f, b in plan["boxes"][t].items()
                        if lo <= f <= hi and b.get("generated", 0) == 0)
    return out


def blur_scores(video_path, plan: dict, keyframes: dict[int, list[int]],
                expected_frames: int | None = None) -> dict[tuple[int, int], float]:
    """{(tid, fid): Laplacian variance of the grey crop inside the box}.

    One sequential decode (cv2.VideoCapture, native frame order; never seek
    by timestamp - rule 6). Reads up to the last needed frame. When
    `expected_frames` is given and the video ends before the last needed
    frame, raises: a frame-count mismatch is reported, never rescaled.
    """
    import cv2
    import numpy as np

    needed: dict[int, list[int]] = {}
    for t, fids in keyframes.items():
        for f in fids:
            needed.setdefault(f, []).append(t)
    if not needed:
        return {}
    last = max(needed)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video_path}")
    out: dict[tuple[int, int], float] = {}
    fid = -1
    try:
        while fid < last:
            ok, frame = cap.read()
            if not ok:
                break
            fid += 1
            if fid not in needed:
                continue
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H, W = grey.shape
            for t in needed[fid]:
                b = plan["boxes"][t][fid]["bbox"]
                x0, y0 = max(0, int(b["xmin"])), max(0, int(b["ymin"]))
                x1, y1 = min(W - 1, int(b["xmax"])), min(H - 1, int(b["ymax"]))
                crop = grey[y0:y1 + 1, x0:x1 + 1]
                if crop.size < 4:
                    out[(t, fid)] = 0.0
                    continue
                lap = cv2.Laplacian(crop, cv2.CV_64F)
                out[(t, fid)] = float(np.var(lap))
    finally:
        cap.release()
    if fid < last:
        raise RuntimeError(
            f"{video_path}: decoded {fid + 1} frames but keyframe {last} is "
            f"needed (annotation frame_count "
            f"{expected_frames if expected_frames is not None else '?'}) - "
            f"rule 6: report, do not rescale")
    return out


def keyframe_metrics(plan: dict, thr: Thresholds,
                     blur: dict[tuple[int, int], float] | None = None) -> dict:
    """{tid: {fid: {edge, overlap, motion, rel_size, blur, flags}}} for every
    human keyframe in the tid span. `blur` None -> the blur flag is skipped
    (recorded as blur=None)."""
    W, H = plan["W"], plan["H"]
    kf = human_keyframes(plan)
    out: dict[int, dict[int, dict]] = {}
    for t, fids in kf.items():
        boxes = plan["boxes"][t]
        all_fids = sorted(boxes)
        areas = [box_area(boxes[f]["bbox"]) for f in fids]
        med_area = st.median(areas) if areas else 0.0
        blur_vals = [blur[(t, f)] for f in fids
                     if blur is not None and (t, f) in blur]
        med_blur = st.median(blur_vals) if blur_vals else None

        def coverage(f: int) -> float:
            b = boxes[f]["bbox"]
            return max((covered_fraction(b, plan["boxes"][o][f]["bbox"])
                        for o in plan["tids"]
                        if o != t and f in plan["boxes"][o]), default=0.0)

        med_overlap = st.median([coverage(f) for f in fids]) if fids else 0.0
        rows = {}
        for f in fids:
            b = boxes[f]["bbox"]
            edge = edge_touches(b, W, H, thr.edge_tol_px)
            overlap = coverage(f)
            motion = motion_at(boxes, all_fids, f)
            rel_size = box_area(b) / med_area if med_area else 1.0
            bl = blur.get((t, f)) if blur is not None else None
            flags = []
            if edge > thr.edge_max_touches or (
                    edge > 0 and rel_size < thr.edge_size_rel):
                flags.append("edge")
            if overlap > thr.overlap_max or (
                    overlap > med_overlap + thr.overlap_rel_delta):
                flags.append("overlap")
            if motion is not None and motion > thr.motion_max:
                flags.append("motion")
            if rel_size < thr.size_min_rel:
                flags.append("small")
            if bl is not None and (bl < thr.blur_abs_min or
                                   (med_blur is not None
                                    and bl < thr.blur_rel_min * med_blur)):
                flags.append("blur")
            rows[f] = dict(edge=edge, overlap=round(overlap, 4),
                           overlap_median=round(med_overlap, 4),
                           motion=None if motion is None else round(motion, 5),
                           rel_size=round(rel_size, 4),
                           blur=None if bl is None else round(bl, 2),
                           flags=flags)
        out[t] = rows
    return out


# -- policy -------------------------------------------------------------------

def _thin(cands: list[int], ref: int, max_anchors: int) -> list[int]:
    """plan_clip's thinning (research code): ref + first + last
    always kept, the rest spread evenly."""
    if max_anchors and len(cands) > max_anchors:
        step = (len(cands) - 1) / (max_anchors - 1)
        return sorted({cands[round(i * step)] for i in range(max_anchors)}
                      | {ref})
    return list(cands)


def _fill_gaps(plan: dict, t: int, rows: dict, kept: list[int], max_gap: int,
               gap_fill: str) -> tuple[list[int], dict]:
    """Add anchors until no stretch of the object's span is longer than `max_gap`
    frames without one: consecutive anchors, the span start to the first anchor, and
    the last anchor to the span end are each at most `max_gap` apart. Each fill is the
    human keyframe with the fewest gate flags nearest the middle of the largest
    remaining gap (flags are recorded, not enforced: coverage wins over the gate here);
    with `gap_fill == "any"` a tracker box is used where no human keyframe lies in the
    gap. Returns (anchors, {fid: {source, flags, gap}}). `max_gap` 0 disables the rule.

    Why: with the gate alone, an object that keeps 1-3 of its keyframes loses the mask
    for hundreds of frames (SAM's memory says "not present" until the next anchor),
    and 1,353 of the 2,265 frames lost on the 20 VidSTG-val review clips were beyond the
    first or last kept anchor (research run of 2026-09-30).
    """
    assert gap_fill in ("human", "any"), gap_fill
    anchors = sorted(kept)
    fills: dict[int, dict] = {}
    if not max_gap or not anchors:
        return anchors, fills
    lo, hi = plan["spans"][t]
    boxed = sorted(f for f in plan["boxes"][t] if lo <= f <= hi)
    unfillable: set[tuple[int, int]] = set()
    while len(anchors) < len(boxed):
        pts = [lo] + anchors + [hi]
        segs = sorted({(a, b) for a, b in zip(pts, pts[1:]) if b - a > max_gap},
                      key=lambda s: s[0] - s[1])
        segs = [s for s in segs if s not in unfillable]
        if not segs:
            break
        a, b = segs[0]
        lo_inc, hi_inc = a == lo and a not in anchors, b == hi and b not in anchors
        cands = [f for f in boxed
                 if (a <= f if lo_inc else a < f) and (f <= b if hi_inc else f < b)]
        mid = (a + b) / 2
        human = [f for f in cands if f in rows]
        if human:
            f = min(human, key=lambda x: (len(rows[x]["flags"]), abs(x - mid), x))
            fills[f] = dict(source="human", flags=list(rows[f]["flags"]), gap=[a, b])
        elif gap_fill == "any" and cands:
            f = min(cands, key=lambda x: (abs(x - mid), x))
            fills[f] = dict(source="tracker", flags=[], gap=[a, b])
        else:
            unfillable.add((a, b))
            continue
        anchors = sorted(set(anchors) | {f})
    return anchors, fills


def apply_hq_policy(plan: dict, metrics: dict, thr: Thresholds,
                    max_anchors: int = 16,
                    fallback: str = "least-flagged") -> dict:
    """Return a copy of `plan` whose anchors are the quality-gated keyframes.

    For each object: candidates = human keyframes with no flags; the reference
    anchor is chosen by plan_clip's rule (min max-IoU vs the other objects at
    that frame, tie earliest) over the candidates; the rest are thinned to
    max_anchors exactly as plan_clip does. Objects the baseline already
    refused (no human keyframe) stay refused. `plan["anchor_mode"]` becomes
    "hq" and `plan["refusals"][tid]` names the reason for a refused object.
    """
    assert fallback in FALLBACKS, fallback
    p = dict(plan)
    p["anchors"] = dict(plan["anchors"])
    p["refusals"] = dict(plan.get("refusals", {}))
    p["anchor_mode"] = "hq"
    p["hq_thresholds"] = thr.as_dict()
    for t in plan["tids"]:
        base = plan["anchors"][t]
        if base is None:
            p["refusals"].setdefault(t, "no_human_keyframe")
            continue
        rows = metrics[t]
        human = sorted(rows)
        clean = [f for f in human if not rows[f]["flags"]]
        used_fallback = False
        flags_ignored: list[str] = []
        if clean:
            cands = clean
        elif fallback == "refuse":
            p["anchors"][t] = None
            p["refusals"][t] = "no_hq_anchor"
            continue
        else:
            k = min(len(rows[f]["flags"]) for f in human)
            cands = [f for f in human if len(rows[f]["flags"]) == k]
            used_fallback = True
            flags_ignored = sorted({fl for f in cands for fl in rows[f]["flags"]})

        def max_iou(f: int) -> float:
            return max((box_iou(plan["boxes"][t][f]["bbox"],
                                plan["boxes"][o][f]["bbox"])
                        for o in plan["tids"]
                        if o != t and f in plan["boxes"][o]), default=0.0)

        ref = min(cands, key=lambda f: (max_iou(f), f))
        kept = _thin(cands, ref, max_anchors)
        edge_keep = ([f for f in (human[0], human[-1]) if f not in kept]
                     if thr.keep_span_edges else [])
        kept = sorted(set(kept) | set(edge_keep))
        kept, fills = _fill_gaps(plan, t, rows, kept, thr.max_gap, thr.gap_fill)
        dropped = {f: (rows[f]["flags"] or ["thinned"])
                   for f in human if f not in kept}
        by_reason: dict[str, int] = {}
        for fl in dropped.values():
            for r in fl:
                by_reason[r] = by_reason.get(r, 0) + 1
        p["anchors"][t] = dict(
            ref=ref, ref_max_iou=max_iou(ref), fids=kept,
            n_human=len(human), anchor_mode="hq",
            n_tracker_anchors=sum(v["source"] == "tracker" for v in fills.values()),
            hq=dict(n_candidates=len(human), n_clean=len(clean),
                    n_kept=len(kept), fallback=used_fallback,
                    flags_ignored=flags_ignored,
                    gap_fills={str(f): v for f, v in sorted(fills.items())},
                    n_gap_fills=len(fills),
                    edge_keep=sorted(set(edge_keep)),
                    dropped={str(f): v for f, v in sorted(dropped.items())},
                    dropped_by_reason=by_reason,
                    baseline_fids=list(base["fids"]),
                    thresholds=thr.as_dict()))
    return p


def human_gap_plan(plan: dict, max_gap: int, gap_fill: str = "human") -> dict:
    """Ablation of the gate: the plain human-keyframe plan (plan_clip, cap 16) with the
    same coverage rule applied and no quality gate — every human keyframe counts as
    clean, so a fill is simply the human keyframe nearest the middle of the largest gap
    (or a tracker box with gap_fill "any"). Answers whether the gate adds anything over
    consistent spacing alone. Records the fills like the hq policy does."""
    p = dict(plan)
    p["anchors"] = dict(plan["anchors"])
    p["anchor_mode"] = "human_gap"
    for t in plan["tids"]:
        a = plan["anchors"][t]
        if a is None:
            continue
        rows = {f: {"flags": []} for f in human_keyframes(plan).get(t, [])}
        kept, fills = _fill_gaps(plan, t, rows, list(a["fids"]), max_gap, gap_fill)
        p["anchors"][t] = dict(a, fids=kept, anchor_mode="human_gap",
                               n_tracker_anchors=sum(v["source"] == "tracker" for v in fills.values()),
                               gap=dict(max_gap=max_gap, gap_fill=gap_fill, n_gap_fills=len(fills),
                                        gap_fills={str(f): v for f, v in sorted(fills.items())},
                                        baseline_fids=list(a["fids"])))
    return p


def hq_plan_for(plan: dict, video_path, thr: Thresholds | None = None,
                max_anchors: int = 16, fallback: str = "least-flagged",
                with_blur: bool = True) -> tuple[dict, dict]:
    """plan_clip output + video -> (gated plan, metrics). One call for the
    runner, the dry run and the plan report so they cannot disagree."""
    thr = thr or Thresholds()
    kf = human_keyframes(plan)
    blur = (blur_scores(video_path, plan, kf, plan.get("frame_count"))
            if with_blur and video_path is not None else None)
    metrics = keyframe_metrics(plan, thr, blur)
    return apply_hq_policy(plan, metrics, thr, max_anchors, fallback), metrics


def describe(plan: dict, t: int) -> str:
    """One line per object for print_plan and the dry run."""
    a = plan["anchors"].get(t)
    if a is None:
        return f"REFUSED ({plan.get('refusals', {}).get(t, 'no_human_keyframe')})"
    h = a.get("hq")
    if not h:
        return f"{len(a['fids'])} anchors (policy {a.get('anchor_mode', 'human')})"
    reasons = ", ".join(f"{k}x{v}" for k, v in sorted(h["dropped_by_reason"].items()))
    fb = (" · FALLBACK least-flagged, ignoring " + "/".join(h["flags_ignored"])
          if h["fallback"] else "")
    fills = h.get("gap_fills") or {}
    ek = h.get("edge_keep") or []
    fb += f" · span edges kept: {ek}" if ek else ""
    gf = (f" · +{len(fills)} gap fills (max gap {h['thresholds'].get('max_gap')}: "
          + ", ".join(f"{f}{'t' if v['source'] == 'tracker' else ''}" for f, v in fills.items()) + ")"
          if fills else "")
    return (f"{h['n_kept']} anchors kept of {h['n_clean']} clean / "
            f"{h['n_candidates']} human keyframes · dropped: {reasons or 'none'}"
            f"{fb}{gf} · ref {a['ref']}")
