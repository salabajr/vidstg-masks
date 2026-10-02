"""Merge the forward and the backward pass of one clip into one set of masks.

Both passes prompt SAM with the same boxes; the forward pass propagates from the span start,
the backward pass from the span end (`sam_session.run_session(direction=...)`). Near an
anchor the pass that has just left it holds the fresher memory, so each pass fails on
different frames. The merge keeps one mask per (object, frame) and writes the rule it applied
into the record (`prompt_payload.merge`), so no choice is silent:

  neither pass has a mask           -> refusal (empty_mask)                       both_empty
  forward only                      -> the forward mask                           forward_only
  backward only                     -> the backward mask                          backward_only
  both, IoU >= agree_iou            -> the forward mask (the passes agree)        agree
  both, dispute_iou <= IoU < agree  -> the mask with the larger share of its pixels inside the
                                       VidOR box; equal -> the larger mask         tiebreak
  both, IoU < dispute_iou           -> the same choice, flagged disputed           disputed
                                       (refuse_disputed: a refusal, disputed_mask) disputed_refused

The forward pass is kept compressed (COCO RLE) while the backward pass runs, so a clip costs
the memory of one pass plus its RLE strings. No torch here: the merge is numpy on the CPU.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from pycocotools import mask as mask_util

RULES = ("both_empty", "forward_only", "backward_only", "agree", "tiebreak", "disputed",
         "disputed_refused")
DEFAULT_AGREE_IOU = 0.7
DEFAULT_DISPUTE_IOU = 0.3


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
    """Share of the mask's pixels inside the (inclusive) VidOR box; None without a box."""
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


def _pick(f: dict, b: dict) -> str:
    """Tie-break: larger inside share, then larger area, then forward."""
    kf = (f["inside"] if f["inside"] is not None else -1.0, f["area"], 1)
    kb = (b["inside"] if b["inside"] is not None else -1.0, b["area"], 0)
    return "forward" if kf >= kb else "backward"


def merge_frame(fwd, bwd, bbox: dict | None, agree_iou: float = DEFAULT_AGREE_IOU,
                dispute_iou: float = DEFAULT_DISPUTE_IOU, refuse_disputed: bool = False):
    """One (object, frame). `fwd` / `bwd` are (mask-or-rle, confidence) or None when that
    pass has no mask there. Returns ((mask, confidence) or None, provenance dict)."""
    mf = None if fwd is None else (decode(fwd[0]) if isinstance(fwd[0], dict) else fwd[0])
    mb = None if bwd is None else (decode(bwd[0]) if isinstance(bwd[0], dict) else bwd[0])
    if mf is not None and not mf.any():
        mf = None
    if mb is not None and not mb.any():
        mb = None
    prov = dict(rule=None, source=None, iou=None, disputed=False,
                forward=None if mf is None else _describe(mf, fwd[1], bbox),
                backward=None if mb is None else _describe(mb, bwd[1], bbox),
                agree_iou=agree_iou, dispute_iou=dispute_iou)
    if mf is None and mb is None:
        prov["rule"] = "both_empty"
        return None, prov
    if mb is None:
        prov.update(rule="forward_only", source="forward")
        return (mf, fwd[1]), prov
    if mf is None:
        prov.update(rule="backward_only", source="backward")
        return (mb, bwd[1]), prov
    prov["iou"] = iou(mf, mb)
    if prov["iou"] >= agree_iou:
        prov.update(rule="agree", source="forward")
        return (mf, fwd[1]), prov
    source = _pick(prov["forward"], prov["backward"])
    if prov["iou"] >= dispute_iou:
        prov.update(rule="tiebreak", source=source)
    elif refuse_disputed:
        prov.update(rule="disputed_refused", source=None, disputed=True)
        return None, prov
    else:
        prov.update(rule="disputed", source=source, disputed=True)
    return ((mf, fwd[1]) if source == "forward" else (mb, bwd[1])), prov


def merge_passes(fwd: dict, bwd: dict, plan: dict, agree_iou: float = DEFAULT_AGREE_IOU,
                 dispute_iou: float = DEFAULT_DISPUTE_IOU, refuse_disputed: bool = False):
    """Whole clip. `fwd` is the compressed forward pass ({fid: {tid: (rle, conf)}}), `bwd`
    the backward pass ({fid: {tid: (mask, conf)}}). Walks every (tid, fid) with a VidOR box
    inside the tid span (the frames that get a record). Returns
    (per_frame {fid: {tid: (mask, conf)}} with the chosen masks only,
     prov {(tid, fid): merge provenance}, Counter of rules)."""
    from .anchors import in_span_fids

    merged: dict[int, dict] = {}
    prov: dict[tuple[int, int], dict] = {}
    rules: Counter = Counter()
    for t in plan["tids"]:
        if plan["anchors"][t] is None:
            continue
        for fid in in_span_fids(plan, t):
            f = fwd.get(fid, {}).get(t)
            b = bwd.get(fid, {}).get(t)
            bbox = plan["boxes"][t][fid]["bbox"]
            chosen, p = merge_frame(f, b, bbox, agree_iou, dispute_iou, refuse_disputed)
            prov[(t, fid)] = p
            rules[p["rule"]] += 1
            if chosen is not None:
                merged.setdefault(fid, {})[t] = chosen
    return merged, prov, rules
