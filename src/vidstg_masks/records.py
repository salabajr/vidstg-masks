"""Mask records: provenance fields, refusal codes, COCO RLE, atomic JSONL, validation.

Every record — mask or refusal — carries the full provenance set. A refusal has
rle=None, mask_confidence=None and a reason_code; nothing else differs.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import CHECKPOINT_SHA256, MODEL_NAME, SAM_VERSION, __version__

PROVENANCE_FIELDS = (
    "vid", "tid", "fid", "rle", "model", "checkpoint_hash", "sam_version",
    "prompt_mode", "prompt_payload", "mask_confidence", "box_generated", "box_tracker",
    "split", "code_commit", "created_at",
)
REASON_CODES = (
    "empty_mask",            # SAM returned no pixels at a frame that has a VidOR box
    "no_human_keyframe",     # no generated==0 box inside the tid span: nothing to seed identity
    "no_hq_anchor",          # hq policy, fallback=refuse: no keyframe passed the quality gate
    "frame_count_mismatch",  # decoded frame count != annotation frame_count (rule 6)
    "frame_size_mismatch",   # decoded WxH != annotation width/height
    "decode_black",          # video decodes as black frames under cv2 (VP6F class); transcode
    "too_many_objects",      # > MAX_OBJECTS relation tids in the clip
    "video_missing",         # no file at <video_root>/<video_path>
    "not_tracked",           # the pass never predicted this frame (the span-end frame of a backward pass)
    "disputed_mask",         # --direction both: two real masks that mostly do not overlap (rule 7)
    "passes_conflict",       # --direction both: two objects' only candidates, from different passes, share pixels
    "handed_over_speck",     # --direction both: a mask trimmed under the speck floor with no backward mask to fall back on
    "speck_mask",            # --direction both: the only mask either pass had was under the speck floor
)
DIRECTIONS = ("forward", "backward", "both")          # --direction; a merged record says "bidirectional"
PAYLOAD_DIRECTIONS = ("forward", "backward", "bidirectional")
CLIP_REASON_CODES = ("frame_count_mismatch", "frame_size_mismatch", "decode_black",
                     "too_many_objects", "video_missing")
PROMPT_MODES = ("pvs_box_multianchor", "pvs_box_hqanchor")
# Mirrors anchors.ANCHOR_POLICIES and the schema enum (pinned equal by tests/test_records.py).
ANCHOR_POLICIES = ("human", "human_gap", "hq")
SPLITS = ("train", "val", "test")


def git_commit(repo_dir: Path | None = None) -> str:
    """Short commit of the package checkout, "-dirty" when modified; when the package is
    not inside a git checkout the version string is recorded instead (never null)."""
    repo_dir = repo_dir or Path(__file__).resolve().parent
    try:
        h = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo_dir), "status", "--porcelain", "."],
                               capture_output=True, text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return f"vidstg_masks-{__version__}+nogit"


def base_provenance(checkpoint_hash: str = CHECKPOINT_SHA256, code_commit: str | None = None) -> dict:
    return dict(model=MODEL_NAME, sam_version=SAM_VERSION, checkpoint_hash=checkpoint_hash,
                code_commit=code_commit or git_commit(),
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def make_record(vid: str, tid: int, fid: int, *, box: dict, split: str, prompt_mode: str,
                prompt_payload: dict, base_prov: dict, rle: dict | None = None,
                mask_confidence: float | None = None, reason_code: str | None = None) -> dict:
    """One (vid, tid, fid) record. Exactly one of (rle, reason_code) is set."""
    if (rle is None) == (reason_code is None):
        raise ValueError("a record is either a mask (rle) or a refusal (reason_code)")
    rec = dict(vid=str(vid), tid=int(tid), fid=int(fid), rle=rle,
               model=base_prov["model"], checkpoint_hash=base_prov["checkpoint_hash"],
               sam_version=base_prov["sam_version"], prompt_mode=prompt_mode,
               prompt_payload=prompt_payload, mask_confidence=mask_confidence,
               box_generated=int(box.get("generated", 0)), box_tracker=box.get("tracker", "none"),
               split=split, code_commit=base_prov["code_commit"],
               created_at=base_prov["created_at"])
    if reason_code is not None:
        rec["reason_code"] = reason_code
    return rec


# ── RLE ─────────────────────────────────────────────────────────────────────

def rle_encode(mask: np.ndarray) -> dict:
    """Compressed COCO RLE ({size: [H, W], counts: str}) via pycocotools."""
    from pycocotools import mask as mask_util
    r = mask_util.encode(np.asfortranarray(np.asarray(mask).astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii")
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"]}


def rle_decode(rle: dict) -> np.ndarray:
    from pycocotools import mask as mask_util
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return mask_util.decode({"size": list(rle["size"]), "counts": counts}).astype(bool)


# ── JSONL ───────────────────────────────────────────────────────────────────

def write_jsonl_atomic(path: Path, records) -> int:
    """Write records to <path>.part, fsync, rename. Returns the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.part")
    n = 0
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                n += 1
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return n


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.part")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=1, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ── Validation (hand check mirroring schema/mask_record.schema.json) ────────

def validate_record(rec: dict) -> None:
    """Raise ValueError on the first violation of the record contract."""
    if not isinstance(rec, dict):
        raise ValueError("record must be an object")
    missing = [k for k in PROVENANCE_FIELDS if k not in rec]
    if missing:
        raise ValueError(f"missing provenance fields: {missing}")
    if not isinstance(rec["vid"], str) or not rec["vid"]:
        raise ValueError("vid must be a non-empty string")
    for k in ("tid", "fid"):
        if not isinstance(rec[k], int) or isinstance(rec[k], bool) or rec[k] < 0:
            raise ValueError(f"{k} must be a non-negative integer")
    for k in ("model", "checkpoint_hash", "sam_version", "code_commit", "created_at", "box_tracker"):
        if not isinstance(rec[k], str) or not rec[k]:
            raise ValueError(f"{k} must be a non-empty string")
    if rec["model"] != MODEL_NAME:
        raise ValueError(f"model must be {MODEL_NAME!r}")
    if rec["sam_version"] != SAM_VERSION:
        raise ValueError(f"sam_version must be {SAM_VERSION!r}")
    if len(rec["checkpoint_hash"]) != 64 or any(c not in "0123456789abcdef" for c in rec["checkpoint_hash"]):
        raise ValueError("checkpoint_hash must be a 64-hex sha256")
    if rec["prompt_mode"] not in PROMPT_MODES:
        raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}")
    pp = rec["prompt_payload"]
    if not isinstance(pp, dict) or "anchor_fids" not in pp or "ref_anchor_fid" not in pp \
            or "anchor_policy" not in pp:
        raise ValueError("prompt_payload needs anchor_fids, ref_anchor_fid, anchor_policy")
    if not isinstance(pp["anchor_fids"], list) or not all(isinstance(f, int) for f in pp["anchor_fids"]):
        raise ValueError("prompt_payload.anchor_fids must be a list of ints")
    if pp["anchor_policy"] not in ANCHOR_POLICIES:
        raise ValueError(f"prompt_payload.anchor_policy must be one of {ANCHOR_POLICIES}")
    if pp["anchor_policy"] == "human_gap":
        if "gap" not in pp:
            raise ValueError("prompt_payload.gap is required under anchor_policy human_gap")
        if not isinstance(pp["gap"], dict) and not (pp["gap"] is None and not pp["anchor_fids"]):
            raise ValueError("prompt_payload.gap must be a dict under human_gap (null only for "
                             "an object that was never prompted)")
    cn = pp.get("contained_negatives")
    if cn is not None and not isinstance(cn, dict):
        raise ValueError("prompt_payload.contained_negatives must be a dict (or null for an "
                         "object that was never prompted)")
    if "direction" in pp and pp["direction"] not in PAYLOAD_DIRECTIONS:
        raise ValueError(f"prompt_payload.direction must be one of {PAYLOAD_DIRECTIONS}")
    # a prompted object of a both run says what the merge did on the frame; an object never
    # prompted (a clip-level refusal, no human keyframe) has no merge to report
    if pp.get("direction") == "bidirectional" and pp.get("anchor_fids") and not isinstance(pp.get("merge"), dict):
        raise ValueError("a bidirectional record of a prompted object carries prompt_payload.merge (the rule applied)")
    if rec["box_generated"] not in (0, 1) or isinstance(rec["box_generated"], bool):
        raise ValueError("box_generated must be 0 (human) or 1 (tracker)")
    if rec["split"] not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    reason = rec.get("reason_code")
    if rec["rle"] is None:
        if reason not in REASON_CODES:
            raise ValueError(f"refusal needs reason_code in {REASON_CODES}, got {reason!r}")
        if rec["mask_confidence"] is not None:
            raise ValueError("refusal must have mask_confidence null")
    else:
        if reason is not None:
            raise ValueError("a mask record must not carry a reason_code")
        rle = rec["rle"]
        if (not isinstance(rle, dict) or not isinstance(rle.get("size"), list)
                or len(rle["size"]) != 2 or not all(isinstance(v, int) and v > 0 for v in rle["size"])
                or not isinstance(rle.get("counts"), str) or not rle["counts"]):
            raise ValueError("rle must be {size: [H, W], counts: <compressed COCO string>}")
        c = rec["mask_confidence"]
        if not isinstance(c, (int, float)) or isinstance(c, bool):
            raise ValueError("mask_confidence must be a number for a mask record")
        if not math.isnan(c) and not 0.0 <= c <= 1.0:
            raise ValueError("mask_confidence must lie in [0, 1]")
