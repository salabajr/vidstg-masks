"""Clip planning from annotations alone (CPU): object set, spans, anchor keyframes.

Ported from the research code. Object set = union of subject_tid /
object_tid over all used_relations of the video's VidSTG records. Clip span = earliest
used_segment start to latest used_segment end; per-tid span = clip span intersected with the tid's VidOR box
presence. Anchors are VidOR human keyframes (generated == 0) inside the tid span; the
reference anchor is the human keyframe whose box is most discriminative against the
other selected tids (minimal max box-IoU at the same frame, tie-break earliest fid).
Up to `max_anchors` anchors are kept per object (ref + evenly thinned rest).

anchor_policy:
  "human"      every human keyframe (capped), without the coverage rule: the base plan that
               `plan_clip` builds, and the CLI default before human_gap.
  "human_gap"  the CLI default: the human plan plus the coverage rule of the hq policy and no
               quality gate (anchor_quality.human_gap_plan): no stretch of an object's span
               longer than `max_gap` frames without an anchor; each gap is filled with the
               human keyframe nearest its middle (a tracker box with gap_fill "any" where no
               human keyframe lies in the gap). Needs no video. Same prompt type as "human".
  "hq"         opt-in: the human keyframes filtered by a visibility/occlusion/motion/size/blur
               gate (anchor_quality.py), then the same coverage rule. Needs the decoded video
               for the blur score, so it is applied by `apply_anchor_policy` after
               `plan_clip`, not inside it.

Contained negatives (any policy; `add_contained_negatives`; on by default in the CLI,
`--no-contained-negatives` turns them off): where another relation object's box lies inside an
object's box and is small relative to it, the container gets a negative click at the contained
object's centre at its own anchors and is co-prompted at the contained object's anchors. Ports
of the research code's contained / add_contained_negatives.
"""

from __future__ import annotations

from pathlib import Path

from .datasets import DataError, boxes_for_tid

ANCHOR_POLICIES = ("human", "human_gap", "hq")
# human_gap prompts exactly like human (boxes at human keyframes): same prompt type.
PROMPT_MODES = {"human": "pvs_box_multianchor", "human_gap": "pvs_box_multianchor",
                "hq": "pvs_box_hqanchor"}


def box_iou(a: dict, b: dict) -> float:
    """IoU of two VidOR bbox dicts (inclusive pixel coords)."""
    iw = min(a["xmax"], b["xmax"]) - max(a["xmin"], b["xmin"]) + 1
    ih = min(a["ymax"], b["ymax"]) - max(a["ymin"], b["ymin"]) + 1
    inter = max(0, iw) * max(0, ih)
    area = lambda r: (r["xmax"] - r["xmin"] + 1) * (r["ymax"] - r["ymin"] + 1)
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def plan_clip(vid: str, records: list[dict], ann: dict, split: str,
              max_anchors: int = 16, anchor_policy: str = "human") -> dict:
    """Everything the GPU step needs that comes from annotations alone.

    `records` are VidSTG-shaped dicts with used_relation, used_segment and
    (optionally) subject/objects. Returns the plan dict; `anchors[tid]` is None for a
    tid refused before any GPU work, with the reason in `refusals[tid]`.
    """
    if anchor_policy not in ANCHOR_POLICIES:
        raise ValueError(f"anchor_policy must be one of {ANCHOR_POLICIES}, got {anchor_policy!r}")
    relations: list[tuple] = []
    for r in records:
        u = r["used_relation"]
        trip = (u["subject_tid"], u["predicate"], u["object_tid"])
        if trip not in relations:
            relations.append(trip)
    tids = sorted({t for s, _, o in relations for t in (s, o)})
    seg = (min(r["used_segment"]["begin_fid"] for r in records),
           max(r["used_segment"]["end_fid"] for r in records))
    cats = {o["tid"]: o["category"]
            for r in records for o in r.get("subject/objects", [])}
    boxes = {t: boxes_for_tid(ann, t) for t in tids}
    spans, anchors, refusals = {}, {}, {}
    for t in tids:
        fids = sorted(boxes[t])
        if not fids:
            raise DataError(f"vid {vid} tid {t}: no VidOR boxes for a relation tid")
        span = (max(seg[0], fids[0]), min(seg[1], fids[-1]))
        spans[t] = span
        human = [f for f in fids if span[0] <= f <= span[1]
                 and boxes[t][f].get("generated", 0) == 0]
        if not human:  # cannot anchor without a human box: refuse the tid
            anchors[t] = None
            refusals[t] = "no_human_keyframe"
            continue
        max_iou = lambda f: max(
            (box_iou(boxes[t][f]["bbox"], boxes[o][f]["bbox"])
             for o in tids if o != t and f in boxes[o]), default=0.0)
        ref = min(human, key=lambda f: (max_iou(f), f))
        cands = human
        # Cap conditioning frames per object: prompt-phase VRAM scales with the
        # per-object anchor count. Ref anchor always kept; the rest thinned evenly.
        kept = cands
        if max_anchors and len(cands) > max_anchors:
            step = (len(cands) - 1) / (max_anchors - 1)
            kept = sorted({cands[round(i * step)]
                           for i in range(max_anchors)} | {ref})
        anchors[t] = dict(ref=ref, ref_max_iou=max_iou(ref), fids=kept,
                          n_human=len(human), anchor_policy="human")
    prop_span = (min(s[0] for s in spans.values()),
                 max(s[1] for s in spans.values()))
    return dict(vid=vid, split=split, W=ann["width"], H=ann["height"],
                fps=ann["fps"], frame_count=ann["frame_count"],
                video_path_rel=ann["video_path"], relations=relations,
                tids=tids, cats=cats, seg=seg, boxes=boxes, spans=spans,
                anchors=anchors, refusals=refusals, prop_span=prop_span,
                anchor_policy="human", max_anchors=max_anchors)


def apply_anchor_policy(plan: dict, anchor_policy: str, video_path: Path | None,
                        max_anchors: int = 16, hq_fallback: str = "least-flagged",
                        max_gap: int = 60, gap_fill: str = "human",
                        keep_span_edges: bool = False) -> dict:
    """Return the plan under `anchor_policy`. "human" is the plan as built, returned as is.
    "human_gap" applies the coverage rule alone (`max_gap` frames, 0 disables it; `gap_fill`
    human or any) and needs no video. "hq" runs the keyframe quality gate (anchor_quality.py)
    with the same coverage rule and `keep_span_edges`; it needs the video for the blur score.
    The input plan is never modified."""
    if anchor_policy == "human":
        return plan
    if anchor_policy not in ANCHOR_POLICIES:
        raise ValueError(f"unknown anchor_policy {anchor_policy!r}")
    try:
        from . import anchor_quality
    except ImportError as e:  # pragma: no cover - only when the module was not shipped
        raise NotImplementedError(
            f"anchor_policy {anchor_policy!r} needs vidstg_masks/anchor_quality.py, which is "
            "not present in this build; use --anchor-policy human") from e
    if anchor_policy == "human_gap":
        gap_plan = anchor_quality.human_gap_plan(plan, max_gap, gap_fill)
        gap_plan["anchor_policy"] = "human_gap"
        for a in gap_plan["anchors"].values():   # fresh dicts from human_gap_plan
            if a is not None:
                a["anchor_policy"] = "human_gap"
        return gap_plan
    if video_path is None:
        raise FileNotFoundError(f"vid {plan['vid']}: the hq gate needs the decoded video "
                                f"for the blur score ({plan['video_path_rel']})")
    thr = anchor_quality.Thresholds(max_gap=max_gap, gap_fill=gap_fill,
                                    keep_span_edges=keep_span_edges)
    hq_plan, _metrics = anchor_quality.hq_plan_for(plan, video_path, thr, max_anchors, hq_fallback)
    hq_plan["anchor_policy"] = "hq"
    return hq_plan


def prompts_for_plan(plan: dict) -> dict[int, list[tuple[int, list[float]]]]:
    """{obj_id: [(fid, [x0, y0, x1, y1] relative 0-1), ...]}, reference anchor FIRST, then
    the remaining anchors in fid order. Objects refused at planning time are absent.
    `add_contained_negatives` may then turn entries into (fid, box, pts, labels)."""
    W, H = plan["W"], plan["H"]
    prompts: dict[int, list] = {}
    for t in plan["tids"]:
        a = plan["anchors"][t]
        if a is None:
            continue
        order = [a["ref"]] + [f for f in a["fids"] if f != a["ref"]]
        prompts[t] = [
            (f, [plan["boxes"][t][f]["bbox"]["xmin"] / W,
                 plan["boxes"][t][f]["bbox"]["ymin"] / H,
                 plan["boxes"][t][f]["bbox"]["xmax"] / W,
                 plan["boxes"][t][f]["bbox"]["ymax"] / H]) for f in order]
    return prompts


def unpack_prompt(entry) -> tuple:
    """(fid, rel_box) or (fid, rel_box, rel_pts, labels) -> 4-tuple with rel_pts/labels as
    (possibly empty) lists."""
    if len(entry) == 2:
        return entry[0], entry[1], [], []
    fid, b, pts, labels = entry
    return fid, b, list(pts), list(labels)


def contained(outer: dict, inner: dict, min_inside: float = 0.9, max_rel_area: float = 0.25) -> bool:
    """inner box lies (almost) inside outer and is small relative to it: a held or
    carried object (toy in a baby's lap, cup in a hand, phone at a face)."""
    ix0, iy0 = max(outer["xmin"], inner["xmin"]), max(outer["ymin"], inner["ymin"])
    ix1, iy1 = min(outer["xmax"], inner["xmax"]), min(outer["ymax"], inner["ymax"])
    inter = max(0, ix1 - ix0 + 1) * max(0, iy1 - iy0 + 1)
    a_in = (inner["xmax"] - inner["xmin"] + 1) * (inner["ymax"] - inner["ymin"] + 1)
    a_out = (outer["xmax"] - outer["xmin"] + 1) * (outer["ymax"] - outer["ymin"] + 1)
    return a_in > 0 and inter / a_in >= min_inside and a_in <= max_rel_area * a_out


def add_contained_negatives(plan: dict, prompts: dict) -> dict:
    """Overlap-aware prompts, from the VidOR boxes alone. For every anchor frame f of
    an object A, each other relation object B whose box at f lies inside A's box and is
    small relative to it adds a NEGATIVE click for A at B's box centre ("A is not
    here"). And at every anchor frame of such a B, A is co-prompted at f with its own
    box plus that negative click, so the multiplex layer resolves the shared pixels
    with both objects' prompts on the table. Without this the larger object annexes
    the smaller one between the smaller one's prompts (research finding: a toy in a
    baby's lap, the baby mask covering 3,128 of the toy box's 3,555 px in every arm).
    Entries of `prompts` become (fid, box) or (fid, box, pts, labels) in the same relative
    0-1 coordinates; labels are 0 (negative click). Returns provenance
    {tid: {"neg_clicks": {fid: [[x, y, tid_B], ...]}, "co_prompt_fids": [...],
    "co_prompt_tracker_fids": [...], "n_neg_clicks": n}}; mutates `prompts`."""
    W, H = plan["W"], plan["H"]
    boxes = plan["boxes"]
    prov = {t: dict(neg_clicks={}, co_prompt_fids=[], co_prompt_tracker_fids=[]) for t in prompts}

    def centre(b):
        return [(b["xmin"] + b["xmax"]) / 2 / W, (b["ymin"] + b["ymax"]) / 2 / H]

    def negatives_for(a, f):
        return [(centre(boxes[b][f]["bbox"]), b) for b in plan["tids"]
                if b != a and f in boxes[b] and f in boxes[a]
                and contained(boxes[a][f]["bbox"], boxes[b][f]["bbox"])]

    # 1. negatives at A's own anchors
    for a, entries in prompts.items():
        new = []
        for e in entries:
            f, box, pts, labels = unpack_prompt(e)
            negs = negatives_for(a, f)
            if negs:
                pts, labels = pts + [c for c, _ in negs], labels + [0] * len(negs)
                prov[a]["neg_clicks"][f] = [[*c, b] for c, b in negs]
            new.append((f, box, pts, labels) if pts else (f, box))
        prompts[a] = new
    # 2. co-prompt the container at the contained object's anchors
    for b, entries in list(prompts.items()):
        for e in entries:
            f = unpack_prompt(e)[0]
            for a in plan["tids"]:
                if a == b or a not in prompts or f not in boxes[a] or f not in boxes[b]:
                    continue
                if not contained(boxes[a][f]["bbox"], boxes[b][f]["bbox"]):
                    continue
                if any(unpack_prompt(x)[0] == f for x in prompts[a]):
                    continue                                    # A already prompted at f (with the negative)
                negs = negatives_for(a, f)
                bb = boxes[a][f]["bbox"]
                rel = [bb["xmin"] / W, bb["ymin"] / H, bb["xmax"] / W, bb["ymax"] / H]
                prompts[a].append((f, rel, [c for c, _ in negs], [0] * len(negs)))
                prov[a]["neg_clicks"][f] = [[*c, t] for c, t in negs]
                prov[a]["co_prompt_fids"].append(f)
                if boxes[a][f].get("generated", 0):
                    prov[a]["co_prompt_tracker_fids"].append(f)
    for t in prov:
        prov[t]["co_prompt_fids"].sort(); prov[t]["co_prompt_tracker_fids"].sort()
        prov[t]["n_neg_clicks"] = sum(len(v) for v in prov[t]["neg_clicks"].values())
        prov[t]["neg_clicks"] = {str(f): v for f, v in sorted(prov[t]["neg_clicks"].items())}
    return prov


def summarize_negatives(prov: dict) -> str:
    """One line for the run log and the plan printout."""
    n_neg = sum(v["n_neg_clicks"] for v in prov.values())
    n_co = sum(len(v["co_prompt_fids"]) for v in prov.values())
    n_cot = sum(len(v["co_prompt_tracker_fids"]) for v in prov.values())
    return (f"contained negatives: {n_neg} negative clicks, {n_co} co-prompts "
            f"({n_cot} at tracker boxes)")


def in_span_fids(plan: dict, tid: int) -> list[int]:
    """Frames inside the tid span that carry a VidOR box: exactly the frames that get a
    record (mask or refusal)."""
    span = plan["spans"][tid]
    return [f for f in sorted(plan["boxes"][tid]) if span[0] <= f <= span[1]]


def describe_plan(plan: dict, video: Path | None = None) -> str:
    p = plan
    n_prop = p["prop_span"][1] - p["prop_span"][0] + 1
    lines = [f"== {p['vid']} · {p['W']}x{p['H']} @ {p['fps']:.2f} fps · "
             f"{p['frame_count']} frames · split {p['split']} · policy {p['anchor_policy']}",
             f"   video: {video if video else 'MISSING (' + p['video_path_rel'] + ')'}",
             f"   relations ({len(p['relations'])}): "
             + " | ".join(f"{s} {pr} {o}" for s, pr, o in p["relations"]),
             f"   segment union {list(p['seg'])} -> propagation span "
             f"{list(p['prop_span'])} (~{n_prop} frames)"]
    for t in p["tids"]:
        a, cat = p["anchors"][t], p["cats"].get(t, "?")
        if a is None:
            lines.append(f"   tid {t} ({cat}): span {list(p['spans'][t])} · "
                         f"REFUSED ({p['refusals'].get(t, 'no_human_keyframe')})")
        else:
            hq = a.get("hq")
            gate = (f" · hq: {hq['n_clean']} clean of {hq['n_candidates']}"
                    f"{' (fallback least-flagged)' if hq['fallback'] else ''}" if hq else "")
            gap = a.get("gap")
            if gap:   # human_gap: the coverage rule's fills over the human plan's anchors
                fills = gap.get("gap_fills") or {}
                gate += (f" · gap rule (max gap {gap['max_gap']}, fill {gap['gap_fill']}): "
                         f"+{gap['n_gap_fills']} fills"
                         + (" (" + ", ".join(f"{f}{'t' if v['source'] == 'tracker' else ''}"
                                             for f, v in fills.items()) + ")" if fills else "")
                         + f" over baseline {gap['baseline_fids']}")
            lines.append(f"   tid {t} ({cat}): span {list(p['spans'][t])} · "
                         f"{len(a['fids'])}/{a.get('n_human', len(a['fids']))} anchors · "
                         f"ref fid {a['ref']} (max IoU vs others {a['ref_max_iou']:.3f}){gate}")
    return "\n".join(lines)
