"""Compatibility shims for the vendored sam3 repo (facebookresearch/sam3 @ 96914d2).

Two upstream bugs bite instance-only (pure PVS) sessions; both patches are
required by every PVS runner (smoke, Phase 2, Phase 3):

1. Sam3BasePredictor.start_session unconditionally passes
   `offload_state_to_cpu` (and, when the attribute exists,
   `video_loader_type`) to model.init_state(), but the SAM 3.1 multiplex
   init_state accepts neither → TypeError on every start_session.
   `patch_start_session` applies the same signature-filter pattern
   upstream's own add_prompt uses.

2. Sam3MultiplexTracking._build_sam2_output returns {} whenever a frame has
   no cached VG (detector) outputs — discarding the refined instance masks
   passed to it. Text-prompt sessions always have VG cache so the webdemo
   never hits this; box/point-prompted instance-only sessions lose every
   propagated mask except the prompt frame (verified with an instrumented
   run: tracker masks nonempty with score ≈ 8.4, output empty).
   `patch_build_sam2_output` merges refined masks over whatever cache
   exists, exactly as the function already does when the cache is present.

Remove once upstream fixes these (re-check on any vendor/sam3 update).
"""

import inspect
import time
import uuid


def patch_start_session(predictor) -> None:
    """Replace predictor.start_session with a signature-filtered version."""
    valid = set(inspect.signature(predictor.model.init_state).parameters)

    def start_session(resource_path, session_id=None,
                      offload_video_to_cpu=False, offload_state_to_cpu=False):
        init_kwargs = dict(
            resource_path=resource_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        if hasattr(predictor, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = predictor.async_loading_frames
        if hasattr(predictor, "video_loader_type"):
            init_kwargs["video_loader_type"] = predictor.video_loader_type
        dropped = sorted(set(init_kwargs) - valid)
        init_kwargs = {k: v for k, v in init_kwargs.items() if k in valid}
        if dropped:
            print(f"sam3_compat: dropped init_state kwargs {dropped}")
        inference_state = predictor.model.init_state(**init_kwargs)

        sid = session_id or str(uuid.uuid4())
        predictor._all_inference_states[sid] = {
            "state": inference_state,
            "session_id": sid,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return {"session_id": sid}

    predictor.start_session = start_session


def patch_build_sam2_output() -> None:
    """Fix _build_sam2_output dropping refined masks on VG-cache-less frames."""
    from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTracking

    def _build_sam2_output(self, inference_state, frame_idx,
                           refined_obj_id_to_mask=None):
        obj_id_to_mask = inference_state["cached_frame_outputs"].get(
            frame_idx, {}).copy()
        if refined_obj_id_to_mask is not None:
            for obj_id, refined_mask in refined_obj_id_to_mask.items():
                assert refined_mask is not None, (
                    f"Refined mask data must be provided for obj_id {obj_id}")
                obj_id_to_mask[obj_id] = refined_mask
        return obj_id_to_mask

    Sam3MultiplexTracking._build_sam2_output = _build_sam2_output


_STATE_OFFLOAD = {"enabled": False}


def set_state_offload(enabled: bool) -> None:
    """Store per-frame tracker outputs on CPU for the NEXT sessions.

    The inner tracker's init_state supports offload_state_to_cpu (documented
    upstream: saves GPU memory at ~10-15% fps cost) but _init_new_sam2_state
    never passes it, so long single-object clips accumulate per-frame outputs
    on GPU until OOM (observed: 2,697-frame G1 clip at 24 GB). Enable per
    clip for long segments; leave off otherwise.
    """
    _STATE_OFFLOAD["enabled"] = enabled


def patch_state_offload() -> None:
    from sam3.model.sam3_multiplex_tracking import \
        Sam3MultiplexTrackingWithInteractivity as M

    def _init_new_sam2_state(self, inference_state):
        return self.tracker.init_state(
            cached_features=inference_state["feature_cache"],
            video_height=inference_state["orig_height"],
            video_width=inference_state["orig_width"],
            num_frames=inference_state["num_frames"],
            offload_state_to_cpu=_STATE_OFFLOAD["enabled"],
        )

    M._init_new_sam2_state = _init_new_sam2_state


def apply_pvs_patches(predictor) -> None:
    """All compat patches a PVS (instance-prompt) runner needs."""
    patch_start_session(predictor)
    patch_build_sam2_output()
    patch_state_offload()
    patch_mux_device()


def collect_sam2_scores(predictor, session_id) -> dict:
    """(fid, obj_id) -> per-frame confidence in [0, 1].

    Upstream's output payload drops the per-frame tracker score (constant
    1.0 lands in mask_confidence); the real scores accumulate in
    tracker_metadata["obj_id_to_sam2_score_frame_wise"] during propagation.
    Scores are logits (~8 for confident tracks) mapped through a sigmoid —
    a monotone, UNCALIBRATED confidence; the add_prompt frame stores a
    sentinel 1.0 which we map to confidence 1.0. Call BEFORE close_session.
    """
    import math

    import torch

    fw = predictor._get_session(session_id)["state"]["tracker_metadata"].get(
        "obj_id_to_sam2_score_frame_wise", {})
    out = {}
    for fid, per_obj in fw.items():
        for obj_id, s in per_obj.items():
            v = float(s.detach().float().cpu()) if torch.is_tensor(s) else float(s)
            out[(int(fid), int(obj_id))] = 1.0 if v == 1.0 \
                else 1.0 / (1.0 + math.exp(-v))
    return out


def purge_prompt_buffers(predictor, session_id, keep=None) -> int:
    """Drop detector buffers leaked by scattered add_prompt calls.

    The multiplex detector caches one full-detector output (~150-230 MB)
    per prompted frame in feature_cache["multigpu_buffer"], plus ~16 MB of
    projected fpn features under integer keys of feature_cache itself, and
    eviction only ever pops frame_idx-1 — an invariant of sequential
    propagation that scattered prompt frames never satisfy, so prompt-phase
    entries leak for the session's lifetime (jobs 31584/31602/31603/31606;
    mechanism verified by CPU buffer simulation + the job-31584 timing
    asymmetry). Call after each add_prompt with keep=<current frame> so
    same-frame prompts still hit; call with keep=None before propagation
    (it re-buffers frames itself). Returns entries dropped.
    """
    state = predictor._get_session(session_id)["state"]
    fc = state["feature_cache"]
    dropped = 0
    for k in [k for k in fc if isinstance(k, int) and k != keep]:
        del fc[k]
        dropped += 1
    mb = fc.get("multigpu_buffer", {})
    for k in [k for k in mb if k != keep]:
        del mb[k]
        dropped += 1
    return dropped

def patch_mux_device() -> None:
    """Make `MultiplexState.mux` tolerate a CPU input.

    With `offload_state_to_cpu`, stored `maskmem_features` live on CPU. When a
    prompt lands on a frame that has already been tracked — the repair2 second
    pass, and only that — the multiplex merge rebuilds a features tensor with
    `device=singleton_features_data.device`, i.e. CPU, and hands it to
    `mux`, whose `mux_matrix` is on CUDA:

        result_flat = self.mux_matrix @ x_flat
        RuntimeError: ... mat2 is on cpu, different from other tensors on cuda:0

    Located from the traceback in job 37923
    (video_tracking_multiplex_demo.py:736 -> multiplex_utils.py:400). The fix
    moves the operand to the matrix's device; it is a no-op when offload is
    off, which is every run before repair2.
    """
    from sam3.model.multiplex_utils import MultiplexState

    if getattr(MultiplexState, "_vidstg_mux_patched", False):
        return
    orig = MultiplexState.mux

    def mux(self, x):
        m = getattr(self, "mux_matrix", None)
        if m is not None and x.device != m.device:
            x = x.to(m.device, non_blocking=True)
        return orig(self, x)

    MultiplexState.mux = mux
    MultiplexState._vidstg_mux_patched = True
