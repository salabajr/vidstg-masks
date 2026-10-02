"""SAM 3.1 Object Multiplex session: build the predictor, run one clip.

torch and sam3 are imported inside functions only. The request sequence is the default
path of run_pilot_masks.run_multiplex_session (research repo), kept identical so masks
regenerate byte-for-byte against existing runs:

  start_session(resource_path=<video>, offload_video_to_cpu=True)
  for each object (plan order): add_prompt at its REFERENCE anchor
      points=[[x0, y0], [x1, y1]] (relative 0-1), point_labels=[2, 3]   # box corners
      purge_prompt_buffers(keep=fid)
  for each remaining anchor, frame-major sorted by (fid, obj_id): add_prompt + purge(keep=fid)
  (a 4-tuple entry from anchors.add_contained_negatives appends its clicks to the corners:
   points=corners + pts, point_labels=[2, 3] + labels, in the SAME add_prompt — a second
   add_prompt on one (object, frame) replaces the earlier points rather than appending)
  purge_prompt_buffers(keep=None)
  propagate_in_video(forward, start_frame_index=start, max_frame_num_to_track=end-start+1)
  collect_sam2_scores()  -> per-frame confidence, BEFORE close_session
  close_session          (in a finally: an orphaned session pins its VRAM)
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from . import CHECKPOINT_SHA256
from .anchors import unpack_prompt
from .sam3_compat import collect_sam2_scores, purge_prompt_buffers

# Tracker prompt-encoder cap (max_point_num_in_prompt_enc): points per add_prompt. Beyond it
# the tracker silently keeps only the first 8 and last 8 points, so it is asserted, never hit.
MAX_PROMPT_POINTS = 16
# State-offload rule (set_state_offload): per-frame tracker outputs accumulate on the GPU
# per object as well as per frame; above these a 24 GB card OOMs.
LONG_CLIP_FRAMES = 1500
LONG_CLIP_FRAME_OBJECTS = 2500


def checkpoint_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_checkpoint(path: Path, expected: str = CHECKPOINT_SHA256) -> str:
    actual = checkpoint_sha256(path)
    if actual != expected:
        raise RuntimeError(f"checkpoint {path} sha256 {actual} != pinned {expected}")
    return actual


def needs_state_offload(n_prop_frames: int, n_objects: int) -> bool:
    return (n_prop_frames > LONG_CLIP_FRAMES
            or n_prop_frames * n_objects > LONG_CLIP_FRAME_OBJECTS)


def build_predictor(checkpoint: Path, n_obj: int):
    """Sam3MultiplexVideoPredictor with the PVS compatibility patches applied. Session
    buffers scale with max_num_objects, so it is sized to the clip being run."""
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    from .sam3_compat import apply_pvs_patches

    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3.1 needs a CUDA device")
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(checkpoint), use_fa3=False, max_num_objects=max(1, n_obj))
    apply_pvs_patches(predictor)
    return predictor


def set_offload(enabled: bool) -> None:
    from .sam3_compat import set_state_offload
    set_state_offload(enabled)


def masks_probs_of(outputs) -> dict[int, tuple[np.ndarray, float]]:
    """obj_id -> (binary HxW mask, prob) from a per-frame outputs payload."""
    result: dict[int, tuple[np.ndarray, float]] = {}
    if not (isinstance(outputs, dict) and "out_obj_ids" in outputs):
        return result
    ids = outputs["out_obj_ids"]
    masks = outputs["out_binary_masks"]
    probs = outputs.get("out_probs", None)
    for k, (i, m) in enumerate(zip(list(ids), list(masks))):
        m = m.cpu().numpy() if hasattr(m, "cpu") and hasattr(m, "numpy") else np.asarray(m)
        p = float(probs[k]) if probs is not None and len(probs) > k else float("nan")
        result[int(i)] = (np.squeeze(m).astype(bool), p)
    return result


def run_session(predictor, video_path, prompts: dict[int, list[tuple]],
                start: int, end: int) -> tuple[dict, float]:
    """prompts: {obj_id: [entry, ...]} with the reference anchor FIRST, where an entry is
    (fid, [x0, y0, x1, y1]) or (fid, [x0, y0, x1, y1], pts, labels): pts a list of [x, y] in
    the same relative (0-1) frame coordinates as the box, labels ints in {1: positive click,
    0: negative click} (anchors.add_contained_negatives). A 2-tuple entry sends exactly the
    request it always did. Returns ({fid: {obj_id: (mask, confidence)}}, wall seconds)."""
    t0 = time.perf_counter()
    sid = predictor.handle_request(request=dict(
        type="start_session", resource_path=str(video_path),
        offload_video_to_cpu=True))["session_id"]

    def add(obj_id, fid, b, pts=(), labels=()):
        """Box corners (labels 2/3) and clicks travel in ONE add_prompt per (obj, frame)."""
        pts, labels = list(pts), list(labels)
        if len(pts) != len(labels):
            raise AssertionError(f"obj {obj_id} fid {fid}: {len(pts)} points vs "
                                 f"{len(labels)} labels")
        if not all(l in (0, 1) for l in labels):
            raise AssertionError(f"obj {obj_id} fid {fid}: click labels must be 0/1 (2/3 are "
                                 f"the box corners, added here), got {labels}")
        corners = [[b[0], b[1]], [b[2], b[3]]]
        if len(corners) + len(pts) > MAX_PROMPT_POINTS:
            raise AssertionError(
                f"obj {obj_id} fid {fid}: {len(corners)} box corners + {len(pts)} clicks exceed "
                f"the prompt-encoder cap of {MAX_PROMPT_POINTS} (max_point_num_in_prompt_enc): "
                f"the tracker would silently keep only the first/last {MAX_PROMPT_POINTS // 2}")
        predictor.handle_request(request=dict(
            type="add_prompt", session_id=sid, frame_index=fid,
            points=corners + pts, point_labels=[2, 3] + labels, obj_id=obj_id))
        purge_prompt_buffers(predictor, sid, keep=fid)

    try:
        # Phase 1: create each object at its discriminative reference anchor.
        for obj_id, anchor_list in prompts.items():
            add(obj_id, *unpack_prompt(anchor_list[0]))
        # Phase 2: re-anchors frame-major, so objects sharing a keyframe reuse one
        # detector buffer entry before it is purged. Sort key is (fid, obj_id) only, never
        # the box/click payload (entries mix tuple lengths).
        refinements = [(fid, obj_id, b, pts, labels) for obj_id, al in prompts.items()
                       for fid, b, pts, labels in map(unpack_prompt, al[1:])]
        for fid, obj_id, b, pts, labels in sorted(refinements, key=lambda r: (r[0], r[1])):
            add(obj_id, fid, b, pts, labels)
        purge_prompt_buffers(predictor, sid, keep=None)
        per_frame: dict[int, dict] = {}
        for resp in predictor.handle_stream_request(request=dict(
                type="propagate_in_video", session_id=sid,
                propagation_direction="forward",
                start_frame_index=start,
                max_frame_num_to_track=end - start + 1)):
            per_frame[resp["frame_index"]] = masks_probs_of(resp["outputs"])
        scores = collect_sam2_scores(predictor, sid)
        for fid, objs in per_frame.items():
            for obj_id, (m, _p) in list(objs.items()):
                objs[obj_id] = (m, scores.get((fid, obj_id), float("nan")))
    finally:
        predictor.handle_request(request=dict(type="close_session", session_id=sid))
    return per_frame, time.perf_counter() - t0


def cuda_peak_gb() -> float:
    import torch
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def cuda_reset_peak() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def is_cuda_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(exc)
