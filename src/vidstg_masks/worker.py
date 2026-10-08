"""Worklist construction, exclusive per-video claims, the shard worker, per-clip records.

Layout under CAMPAIGN_ROOT:
  worklist.json          units + counts + roots (built once on a CPU node)
  records/<vid>.jsonl    one record per (tid, fid) with a VidOR box inside the tid span
  runs/<vid>.json        timing / VRAM / refusal counts for a completed clip
  errors/<vid>.json      last failure with attempt count
  claims/<vid>.claim     exclusive lease (heartbeat mtime; stale after 30 min)

Every shard walks the whole worklist from its own offset and processes what nobody has
claimed, so N array tasks balance dynamically. Each clip runs in a SUBPROCESS
(`vidstg-masks process-one`) because SAM 3.1 does not release session VRAM between clips.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .anchors import (PROMPT_MODES, add_contained_negatives, apply_anchor_policy, describe_plan,
                      in_span_fids, plan_clip, prompts_for_plan, summarize_negatives)
from .datasets import (VIDOR_INDEX_PATTERN, DataError, Roots, assert_known_counts,
                       build_vidor_index, load_vidor, load_vidstg, rebase_unit_paths,
                       records_by_vid, relation_tids, resolve_video)
from .records import (DIRECTIONS, base_provenance, make_record, read_jsonl, rle_encode,
                      write_json_atomic, write_jsonl_atomic)
from .video import decode_check

WORKLIST_VERSION = "vidstg-masks-worklist-v1"
MAX_OBJECTS = 16          # multiplex bucket size; above this the clip is refused
CLAIM_STALE_SECONDS = 30 * 60
CLAIM_HEARTBEAT_SECONDS = 30
EXIT_OOM = 3
EXIT_DRAINED = 99
MAX_ATTEMPTS = 2          # OOM: one retry with tracker state offloaded; other errors: none
MAX_INTERRUPTIONS = 5     # a clip killed by a signal (walltime, preemption) is retried this often
# Per-run settings written into every runs/<vid>.json (and printed by the worker) so a reader
# can tell how a campaign was run from its outputs alone.
SETTINGS_KEYS = ("anchor_policy", "max_anchors", "hq_fallback", "max_gap", "gap_fill",
                 "keep_span_edges", "contained_negatives",
                 "direction", "agree_iou", "speck_floor", "speck_ratio", "dispute_score",
                 "dispute_rule", "dispute_winner")

STOP_REQUESTED = False


def run_settings(anchor_policy: str, max_anchors: int, hq_fallback: str = "least-flagged",
                 max_gap: int = 60, gap_fill: str = "human", keep_span_edges: bool = False,
                 contained_negatives: bool = False, direction: str = "both",
                 agree_iou: float = 0.3, speck_floor: int = 20,
                 speck_ratio: float = 0.1, dispute_score: float | None = 0.907,
                 dispute_rule: str = "higher_score", dispute_winner: str = "strong") -> dict:
    return dict(anchor_policy=anchor_policy, max_anchors=max_anchors, hq_fallback=hq_fallback,
                max_gap=max_gap, gap_fill=gap_fill, keep_span_edges=keep_span_edges,
                contained_negatives=contained_negatives, direction=direction,
                agree_iou=agree_iou, speck_floor=speck_floor, speck_ratio=speck_ratio,
                dispute_score=dispute_score, dispute_rule=dispute_rule, dispute_winner=dispute_winner)


def request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[drain] signal {signum}: finishing the active clip, then exiting 99", flush=True)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


# ── Worklist ────────────────────────────────────────────────────────────────

def read_vids_file(path: Path) -> list[str]:
    """One vid per line, first whitespace token; `#` lines are comments."""
    vids = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        vids.append(s.split()[0])
    return vids


def unit_records(unit: dict) -> list[dict]:
    """VidSTG-shaped records rebuilt from a worklist unit (one per relation, all sharing the
    unit's segment hull), so plan_clip reproduces the plan without the VidSTG files."""
    so = [{"tid": int(t), "category": c} for t, c in unit["cats"].items()]
    return [{"used_relation": {"subject_tid": s, "predicate": p, "object_tid": o},
             "used_segment": {"begin_fid": unit["segment"][0], "end_fid": unit["segment"][1]},
             "subject/objects": so} for s, p, o in unit["relations"]]


def build_worklist(roots: Roots, split: str = "all", vids: list[str] | None = None,
                   limit: int | None = None, assert_counts: bool = True) -> dict:
    """Select the videos of the requested VidSTG split(s) and freeze, per video, the
    relations, the annotation path and the video path. With `assert_counts` (the default)
    the loaded annotation set must carry the release counts, so a partial or altered copy
    of the dataset is caught before any GPU work."""
    splits = ("train", "val", "test") if split == "all" else (split,)
    records = load_vidstg(roots, splits)
    index = build_vidor_index(roots)
    if assert_counts:
        assert_known_counts(records, splits, index, roots)
    by_vid = records_by_vid(records)
    if vids is not None:
        wanted = [v for v in vids if v in by_vid]
        unknown = sorted(set(vids) - set(by_vid))
        if unknown:
            print(f"[worklist] {len(unknown)} requested vids have no VidSTG record in "
                  f"{'/'.join(splits)}: {unknown[:10]}{' ...' if len(unknown) > 10 else ''}",
                  file=sys.stderr)
    else:
        wanted = sorted(by_vid)
    if limit:
        wanted = wanted[:limit]
    units, missing_ann = [], []
    for vid in wanted:
        recs = by_vid[vid]
        if vid not in index:
            missing_ann.append(vid)
            continue
        ann = load_vidor(index[vid])
        relations = []
        for r in recs:
            u = r["used_relation"]
            trip = [u["subject_tid"], u["predicate"], u["object_tid"]]
            if trip not in relations:
                relations.append(trip)
        seg = [min(r["used_segment"]["begin_fid"] for r in recs),
               max(r["used_segment"]["end_fid"] for r in recs)]
        cats = {str(o["tid"]): o["category"] for r in recs for o in r["subject/objects"]}
        vsplits = sorted({r["vidstg_split"] for r in recs})
        video = resolve_video(roots, ann)
        units.append(dict(
            vid=vid, vidstg_split="+".join(vsplits), n_tids=len(relation_tids(recs)),
            n_frames=ann["frame_count"], n_records=len(recs),
            segment=seg, relations=relations, cats=cats,
            vidor_ann=str(index[vid]), video_path=str(video) if video else None))
    if missing_ann:
        raise DataError(f"{len(missing_ann)} VidSTG videos have no VidOR annotation under "
                        f"{roots.vidor_ann_root} ({len(index):,} files matched "
                        f"{VIDOR_INDEX_PATTERN}): {missing_ann[:10]}")
    counts = {
        "units": len(units),
        "by_split": dict(Counter(u["vidstg_split"] for u in units)),
        "video_missing": sum(1 for u in units if u["video_path"] is None),
        "too_many_objects": sum(1 for u in units if u["n_tids"] > MAX_OBJECTS),
        "relation_tids": sum(u["n_tids"] for u in units),
        "frames": sum(u["n_frames"] for u in units),
        "prop_frame_objects": sum((u["segment"][1] - u["segment"][0] + 1) * u["n_tids"]
                                  for u in units),
    }
    return dict(version=WORKLIST_VERSION, split=split,
                created_at=_now(),
                roots=roots.to_dict(), counts=counts, units=units)


def save_worklist(path: Path, worklist: dict) -> None:
    write_json_atomic(path, worklist)


def load_worklist(path: Path) -> dict:
    wl = json.loads(Path(path).read_text())
    if wl.get("version") != WORKLIST_VERSION:
        raise DataError(f"{path}: worklist version {wl.get('version')!r} != {WORKLIST_VERSION!r}")
    return wl


def shard_order(units: list, shard_index: int, shard_count: int) -> list:
    """The whole worklist rotated so this shard starts at its own block and then walks
    the other shards' blocks — claims make the overlap safe (dynamic balancing)."""
    if not units:
        return []
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= index < shard_count")
    b = -(-len(units) // shard_count)
    off = (shard_index * b) % len(units)
    return units[off:] + units[:off]


# ── Claims ──────────────────────────────────────────────────────────────────

@contextmanager
def claim(path: Path, owner: str, stale_after: float = CLAIM_STALE_SECONDS,
          heartbeat: float = CLAIM_HEARTBEAT_SECONDS):
    """Exclusive lease file (O_EXCL). Yields True when acquired. A claim whose mtime is
    older than `stale_after` belongs to a dead worker and is taken over."""
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    try:
        st = path.stat()
    except FileNotFoundError:
        pass
    else:
        same_owner = False
        try:
            same_owner = json.loads(path.read_text()).get("owner") == owner
        except (OSError, ValueError, AttributeError):
            pass
        if same_owner or now - st.st_mtime > stale_after:
            try:
                path.replace(path.with_suffix(path.suffix + f".stale-{int(now)}-{os.getpid()}"))
            except FileNotFoundError:
                pass
        else:
            yield False
            return
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        yield False
        return
    with os.fdopen(fd, "w") as f:
        json.dump({"owner": owner, "pid": os.getpid(), "claimed_at": now}, f)
        f.write("\n")
    stopped = threading.Event()

    def beat():
        while not stopped.wait(heartbeat):
            try:
                os.utime(path, None)
            except FileNotFoundError:
                return

    t = threading.Thread(target=beat, name="claim-heartbeat", daemon=True)
    t.start()
    try:
        yield True
    finally:
        stopped.set()
        t.join(timeout=max(1, heartbeat))
        try:
            current = json.loads(path.read_text())
        except (OSError, ValueError):
            current = {}
        if current.get("owner") == owner:
            path.unlink(missing_ok=True)


# ── Per-clip work (CPU planning + record export) ────────────────────────────

def plan_unit(unit: dict, roots: Roots, max_anchors: int, anchor_policy: str,
              hq_fallback: str = "least-flagged", apply_policy: bool = True,
              max_gap: int = 60, gap_fill: str = "human",
              keep_span_edges: bool = False) -> tuple[dict, Path | None]:
    """Human plan for a worklist unit (+ resolved video). With `apply_policy` the requested
    anchor policy is applied too (`max_gap`, `gap_fill`, `keep_span_edges` are its coverage
    rule settings); run_clip applies it itself AFTER the decode pre-check, because the hq
    gate reads the video."""
    ann = load_vidor(Path(unit["vidor_ann"]))
    video = Path(unit["video_path"]) if unit.get("video_path") else resolve_video(roots, ann)
    if video is not None and not video.exists():
        video = resolve_video(roots, ann)
    plan = plan_clip(unit["vid"], unit_records(unit), ann, unit["vidstg_split"],
                     max_anchors, "human")
    if apply_policy:
        plan = apply_anchor_policy(plan, anchor_policy, video, max_anchors, hq_fallback,
                                   max_gap, gap_fill, keep_span_edges)
    return plan, video


def precheck_clip(plan: dict, video: Path | None) -> tuple[str | None, dict]:
    """CPU gate before any GPU work. Returns (clip-level reason code or None, details)."""
    if len(plan["tids"]) > MAX_OBJECTS:
        return "too_many_objects", {"n_tids": len(plan["tids"])}
    if video is None:
        return "video_missing", {"video_path_rel": plan["video_path_rel"]}
    chk = decode_check(video, plan["frame_count"], (plan["W"], plan["H"]))
    if not chk["frame_count_ok"]:
        return "frame_count_mismatch", chk
    if not chk["size_ok"]:
        return "frame_size_mismatch", chk
    if chk["black"]:
        return "decode_black", chk
    return None, chk


def clip_records(plan: dict, per_frame: dict | None, base_prov: dict,
                 clip_reason: str | None = None, neg_prov: dict | None = None,
                 merge_prov: dict | None = None, direction: str = "forward") -> tuple[list[dict], Counter]:
    """All records for a clip: one per (tid, fid) with a VidOR box inside the tid span.
    `clip_reason` refuses every frame of every tid (video-level defect: no prompt was ever
    issued, so the payload carries no anchors); otherwise a tid without anchors is refused
    with its planning reason, a frame the pass never predicted with `not_tracked` and a
    frame with an empty mask with `empty_mask`.
    `neg_prov` (anchors.add_contained_negatives) goes into each object's payload as
    `contained_negatives`, and its co-prompt frames widen the payload's `anchor_fids`.
    `direction` is recorded in the payload; with `merge_prov` ({(tid, fid): merge.merge_frame
    provenance}, a `--direction both` run) the payload says `bidirectional` and carries the
    frame's `merge` block; a frame the merge refused carries its `refused_reason`, and a
    frame with nothing real in either pass `empty_mask` (both empty) or `speck_mask`."""
    prompt_mode = PROMPT_MODES[plan["anchor_policy"]]
    out, refusals = [], Counter()
    for t in plan["tids"]:
        a = plan["anchors"][t] if clip_reason is None else None
        payload = dict(anchor_fids=list(a["fids"]) if a else [],
                       ref_anchor_fid=a["ref"] if a else None,
                       anchor_policy=plan["anchor_policy"])
        # a forward pass keeps the payload exactly as it always was (records regenerate
        # byte for byte); any other direction says so
        pay_dir = "bidirectional" if (merge_prov is not None or direction == "both") else direction
        if pay_dir != "forward":
            payload["direction"] = pay_dir
        if plan["anchor_policy"] == "hq":
            payload["hq"] = (a or {}).get("hq")
        elif plan["anchor_policy"] == "human_gap":
            payload["gap"] = (a or {}).get("gap")
        if neg_prov is not None:
            payload["contained_negatives"] = neg_prov.get(t)
            payload["anchor_fids"] = sorted({*payload["anchor_fids"],
                                             *neg_prov.get(t, {}).get("co_prompt_fids", [])})
        tid_reason = plan["refusals"].get(t, "no_human_keyframe") if plan["anchors"][t] is None else None
        for fid in in_span_fids(plan, t):
            box = plan["boxes"][t][fid]
            mp = merge_prov.get((t, fid)) if merge_prov is not None else None
            common = dict(box=box, split=plan["split"], prompt_mode=prompt_mode,
                          prompt_payload=dict(payload, merge=mp) if merge_prov is not None else payload,
                          base_prov=base_prov)
            if clip_reason is not None:
                reason = clip_reason
            elif tid_reason is not None:
                reason = tid_reason
            elif mp is not None and mp["decision"] == "refused":
                reason = mp["refused_reason"]
            elif mp is not None and mp["decision"] == "none":
                reason = "empty_mask" if mp["forward"] is None and mp["backward"] is None else "speck_mask"
            else:
                got = (per_frame or {}).get(fid, {}).get(t)
                if got is None and merge_prov is None and per_frame and fid not in per_frame:
                    reason = "not_tracked"       # the pass never predicted this frame
                elif got is None or not got[0].any():
                    reason = "empty_mask"
                else:
                    out.append(make_record(plan["vid"], t, fid, rle=rle_encode(got[0]),
                                           mask_confidence=got[1], **common))
                    continue
            refusals[reason] += 1
            out.append(make_record(plan["vid"], t, fid, reason_code=reason, **common))
    return out, refusals


def campaign_dirs(campaign_root: Path) -> dict[str, Path]:
    return {k: campaign_root / k for k in ("records", "runs", "errors", "claims")}


def refuse_clip(unit: dict, roots: Roots, campaign_root: Path, reason: str,
                base_prov: dict, max_anchors: int, anchor_policy: str, details: dict | None = None,
                plan: dict | None = None, settings: dict | None = None) -> dict:
    """Write refusal records for the whole clip (no GPU, no prompt). The records name the
    requested anchor policy; their payload has no anchors because nothing was prompted.
    `settings` (run_settings) is recorded in runs/<vid>.json like a completed clip's."""
    if plan is None:
        plan, _ = plan_unit(unit, roots, max_anchors, anchor_policy, apply_policy=False)
    plan = dict(plan, anchor_policy=anchor_policy)
    recs, refusals = clip_records(plan, None, base_prov, clip_reason=reason,
                                  direction=(settings or {}).get("direction", "forward"))
    d = campaign_dirs(campaign_root)
    write_jsonl_atomic(d["records"] / f"{unit['vid']}.jsonl", recs)
    run = dict(vid=unit["vid"], status="refused", reason=reason, details=details or {},
               masks=0, refusals=dict(refusals), n_records=len(recs), gpu=False,
               anchor_policy=anchor_policy, max_anchors=max_anchors,
               finished_at=_now())
    run.update(settings or {})
    write_json_atomic(d["runs"] / f"{unit['vid']}.json", run)
    return run


def run_clip(unit: dict, roots: Roots, campaign_root: Path, checkpoint: Path,
             anchor_policy: str, max_anchors: int, force_offload: bool = False,
             hq_fallback: str = "least-flagged", verbose: bool = True,
             checkpoint_hash: str | None = None, max_gap: int = 60, gap_fill: str = "human",
             keep_span_edges: bool = False, contained_negatives: bool = False,
             direction: str = "both", agree_iou: float = 0.3, speck_floor: int = 20,
             speck_ratio: float = 0.1, dispute_score: float | None = 0.907,
             dispute_rule: str = "higher_score", dispute_winner: str = "strong") -> dict:
    """One clip end to end inside the current process: plan, precheck, SAM, records.
    `direction`: forward or backward, one pass; both = a forward and a backward session
    with the same prompts, merged by merge.merge_passes (the pixels rule: `agree_iou`,
    `speck_floor`, `speck_ratio`, and the dispute tie-break `dispute_rule` /
    `dispute_score` / `dispute_winner`; every record then carries what the merge did).
    `checkpoint_hash` is the sha256 the parent shard already verified (so the 3.5 GB file
    is hashed once per shard, not once per clip); without it the file is hashed here.
    `max_gap` / `gap_fill` / `keep_span_edges` are the coverage-rule settings of the
    human_gap and hq policies; `contained_negatives` adds the overlap-aware clicks and
    co-prompts of anchors.add_contained_negatives under any policy."""
    from . import merge as mg
    from . import sam_session as ss

    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    mg.check_dispute_settings(dispute_rule, dispute_score, dispute_winner)
    d = campaign_dirs(campaign_root)
    settings = run_settings(anchor_policy, max_anchors, hq_fallback, max_gap, gap_fill,
                            keep_span_edges, contained_negatives, direction, agree_iou,
                            speck_floor, speck_ratio, dispute_score, dispute_rule, dispute_winner)
    plan, video = plan_unit(unit, roots, max_anchors, anchor_policy, apply_policy=False)
    reason, details = precheck_clip(plan, video)
    if checkpoint_hash is None:
        checkpoint_hash = ss.assert_checkpoint(checkpoint)
    base_prov = base_provenance(checkpoint_hash)
    if reason is not None:
        if verbose:
            print(f"== {unit['vid']}: REFUSED clip: {reason} "
                  f"{ {k: v for k, v in details.items() if k != 'sampled_means'} }", flush=True)
        return refuse_clip(unit, roots, campaign_root, reason, base_prov, max_anchors,
                           anchor_policy, {k: v for k, v in details.items() if k != "sampled_means"},
                           plan=plan, settings=settings)
    # The video decodes as annotated: the hq gate may now read it for the blur score.
    plan = apply_anchor_policy(plan, anchor_policy, video, max_anchors, hq_fallback,
                               max_gap, gap_fill, keep_span_edges)
    if verbose:
        print(describe_plan(plan, video), flush=True)
    prompts = prompts_for_plan(plan)
    neg_prov = None
    if contained_negatives:
        neg_prov = add_contained_negatives(plan, prompts)
        if verbose:
            print(f"   {summarize_negatives(neg_prov)}", flush=True)
    n_prop = plan["prop_span"][1] - plan["prop_span"][0] + 1
    per_frame, wall, vram, offload = {}, 0.0, 0.0, None
    merge_prov, merge_rules, frames_run = None, None, 0
    if prompts:
        offload = force_offload or ss.needs_state_offload(n_prop, len(prompts))
        predictor = ss.build_predictor(checkpoint, len(plan["tids"]))
        ss.set_offload(offload)
        if verbose:
            print(f"   state offload {'ON' if offload else 'off'} "
                  f"({n_prop} frames x {len(prompts)} objects)"
                  f"{' [forced after OOM]' if force_offload else ''}", flush=True)
        ss.cuda_reset_peak()
        s0, s1 = plan["prop_span"]
        if direction == "both":
            # the forward pass is kept as RLE while the backward pass runs; then the merge
            fwd, wall_f = ss.run_session(predictor, video, prompts, s0, s1, "forward")
            fwd_rle = mg.compress(fwd)
            del fwd
            bwd, wall_b = ss.run_session(predictor, video, prompts, s0, s1, "backward")
            wall = wall_f + wall_b
            frames_run = len(fwd_rle) + len(bwd)
            per_frame, merge_prov, merge_rules = mg.merge_passes(
                fwd_rle, bwd, plan, agree_iou, speck_floor, speck_ratio, dispute_score,
                dispute_rule, dispute_winner)
            if verbose:
                print(f"   merge: {dict(merge_rules)}", flush=True)
        else:
            per_frame, wall = ss.run_session(predictor, video, prompts, s0, s1, direction)
            frames_run = len(per_frame)
        vram = ss.cuda_peak_gb()
    recs, refusals = clip_records(plan, per_frame, base_prov, neg_prov=neg_prov,
                                  merge_prov=merge_prov, direction=direction)
    write_jsonl_atomic(d["records"] / f"{unit['vid']}.jsonl", recs)
    run = dict(vid=unit["vid"], status="done", video=str(video), prop_span=list(plan["prop_span"]),
               frames=frames_run, objects=len(prompts), masks=len(recs) - sum(refusals.values()),
               refusals=dict(refusals), n_records=len(recs), gpu=bool(prompts),
               ms_per_frame=1000 * wall / max(1, frames_run), wall_s=wall, vram_gb=vram,
               state_offload=offload, merge=dict(merge_rules) if merge_rules is not None else None,
               **settings,
               decoded_frames=details.get("frame_count"), ann_frames=plan["frame_count"],
               finished_at=_now())
    write_json_atomic(d["runs"] / f"{unit['vid']}.json", run)
    if verbose:
        print(f"   {run['masks']} masks / {sum(refusals.values())} refused · "
              f"{run['ms_per_frame']:.0f} ms/f · {vram:.1f} GB", flush=True)
    return run


def process_one_main(worklist_path: Path, vid: str, campaign_root: Path, checkpoint: Path,
                     anchor_policy: str, max_anchors: int, attempt: int,
                     force_offload: bool, hq_fallback: str = "least-flagged",
                     checkpoint_hash: str | None = None, roots: Roots | None = None,
                     max_gap: int = 60, gap_fill: str = "human", keep_span_edges: bool = False,
                     contained_negatives: bool = False, direction: str = "both",
                     agree_iou: float = 0.3, speck_floor: int = 20,
                     speck_ratio: float = 0.1, dispute_score: float | None = 0.907,
                     dispute_rule: str = "higher_score", dispute_winner: str = "strong") -> int:
    """Entry point of the per-clip subprocess. Writes errors/<vid>.json on failure and
    exits EXIT_OOM for a CUDA OOM so the parent can retry with offload forced on.
    `roots`, when given (--roots-from-env), replaces the roots frozen in the worklist and
    the unit's paths are rebased onto it."""
    from . import sam_session as ss

    wl = load_worklist(worklist_path)
    wl_roots = Roots.from_dict(wl["roots"])
    unit = next((u for u in wl["units"] if u["vid"] == vid), None)
    if unit is None:
        print(f"vid {vid} not in {worklist_path}", file=sys.stderr)
        return 2
    if roots is not None:
        unit = rebase_unit_paths(unit, wl_roots, roots)
    else:
        roots = wl_roots
    d = campaign_dirs(campaign_root)
    try:
        run_clip(unit, roots, campaign_root, checkpoint, anchor_policy, max_anchors,
                 force_offload, hq_fallback, checkpoint_hash=checkpoint_hash,
                 max_gap=max_gap, gap_fill=gap_fill, keep_span_edges=keep_span_edges,
                 contained_negatives=contained_negatives, direction=direction,
                 agree_iou=agree_iou, speck_floor=speck_floor, speck_ratio=speck_ratio,
                 dispute_score=dispute_score, dispute_rule=dispute_rule, dispute_winner=dispute_winner)
        (d["errors"] / f"{vid}.json").unlink(missing_ok=True)
        return 0
    except Exception as e:  # noqa: BLE001 - record, never drop
        oom = ss.is_cuda_oom(e)
        write_json_atomic(d["errors"] / f"{vid}.json", dict(
            vid=vid, attempt=attempt, error_type=type(e).__name__, message=str(e)[:2000],
            cuda_oom=oom, force_offload=force_offload, traceback=traceback.format_exc(),
            failed_at=_now()))
        print(f"[failed] {vid} attempt={attempt} {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_OOM if oom else 1


def subprocess_runner(worklist_path: Path, campaign_root: Path, checkpoint: Path,
                      anchor_policy: str, max_anchors: int, python: str | None = None,
                      hq_fallback: str = "least-flagged", roots: Roots | None = None,
                      max_gap: int = 60, gap_fill: str = "human", keep_span_edges: bool = False,
                      contained_negatives: bool = False, direction: str = "both",
                      agree_iou: float = 0.3, speck_floor: int = 20,
                      speck_ratio: float = 0.1, dispute_score: float | None = 0.907,
                      dispute_rule: str = "higher_score", dispute_winner: str = "strong"):
    """Default clip runner: one fresh Python process per clip. The checkpoint is hashed
    once, before the first clip, and the hash handed to every child. `roots` (from
    --roots-from-env) is passed to the child as explicit root flags; the policy settings
    travel as the same flags `process` took."""
    state: dict[str, str | None] = {"hash": None}

    def run(unit: dict, attempt: int, force_offload: bool) -> int:
        if state["hash"] is None:
            from . import sam_session as ss
            state["hash"] = ss.assert_checkpoint(checkpoint)
            print(f"[worker] checkpoint {checkpoint}: sha256 {state['hash'][:16]}… == pinned",
                  flush=True)
        cmd = [python or sys.executable, "-m", "vidstg_masks.cli", "process-one",
               "--worklist", str(worklist_path), "--vid", unit["vid"],
               "--campaign-root", str(campaign_root), "--checkpoint", str(checkpoint),
               "--checkpoint-hash", state["hash"],
               "--anchor-policy", anchor_policy, "--max-anchors", str(max_anchors),
               "--hq-fallback", hq_fallback, "--max-gap", str(max_gap), "--gap-fill", gap_fill,
               "--attempt", str(attempt)]
        if keep_span_edges:
            cmd.append("--keep-span-edges")
        # sent either way: the child's own default is on, so a shard started with
        # --no-contained-negatives has to say so to every clip
        cmd.append("--contained-negatives" if contained_negatives else "--no-contained-negatives")
        cmd += ["--direction", direction, "--agree-iou", str(agree_iou),
                "--speck-floor", str(speck_floor), "--speck-ratio", str(speck_ratio)]
        cmd += ["--dispute-rule", dispute_rule, "--dispute-winner", dispute_winner]
        if dispute_score is not None:
            cmd += ["--dispute-score", str(dispute_score)]
        if force_offload:
            cmd.append("--force-offload")
        if roots is not None:
            cmd += ["--roots-from-env", "--vidstg-root", str(roots.vidstg_root),
                    "--vidor-ann-root", str(roots.vidor_ann_root),
                    "--vidor-video-root", str(roots.vidor_video_root)]
            if roots.vidor_transcoded_root is not None:
                cmd += ["--vidor-transcoded-root", str(roots.vidor_transcoded_root)]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            p for p in [str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", "")] if p))
        return subprocess.run(cmd, env=env).returncode
    return run


def _error(errors_dir: Path, vid: str) -> dict:
    p = errors_dir / f"{vid}.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {"attempt": 1, "error_type": "unreadable"}


def unit_status(unit: dict, campaign_root: Path, max_attempts: int = MAX_ATTEMPTS) -> dict:
    """done (records exist) / failed (terminal error) / pending. Terminal: an interrupted
    clip (child killed by a signal, or the shard was draining) after MAX_INTERRUPTIONS
    attempts; a CUDA OOM after `max_attempts`; any other error after one attempt."""
    d = campaign_dirs(campaign_root)
    vid = unit["vid"]
    if (d["records"] / f"{vid}.jsonl").is_file():
        run = {}
        rp = d["runs"] / f"{vid}.json"
        if rp.is_file():
            try:
                run = json.loads(rp.read_text())
            except (OSError, ValueError):
                run = {}
        return {"status": "done", "attempts": 0, "run": run}
    err = _error(d["errors"], vid)
    attempts = int(err.get("attempt", 0) or 0)
    if err:
        if err.get("interrupted"):
            terminal = attempts >= MAX_INTERRUPTIONS
        else:
            terminal = attempts >= max_attempts or (attempts >= 1 and not err.get("cuda_oom"))
        if terminal:
            return {"status": "failed", "attempts": attempts, "error": err}
    return {"status": "pending", "attempts": attempts, "error": err}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _refuse_on_cpu(unit: dict, roots: Roots, campaign_root: Path, anchor_policy: str,
                   max_anchors: int, max_attempts: int, base_prov: dict,
                   settings: dict | None = None) -> bool:
    """Refusals that need no GPU (too many objects, video missing): write the refusal
    records and return True; return False when the clip needs SAM."""
    reason = ("too_many_objects" if unit["n_tids"] > MAX_OBJECTS
              else "video_missing" if not unit.get("video_path")
              or not Path(unit["video_path"]).exists() else None)
    if reason is None:
        return False
    vid = unit["vid"]
    try:
        refuse_clip(unit, roots, campaign_root, reason, base_prov, max_anchors, anchor_policy,
                    settings=settings)
        print(f"[refused] {vid} {reason}", flush=True)
    except Exception as e:  # noqa: BLE001 - record, never drop
        write_json_atomic(campaign_dirs(campaign_root)["errors"] / f"{vid}.json", dict(
            vid=vid, attempt=max_attempts, error_type=type(e).__name__, message=str(e)[:2000],
            cuda_oom=False, traceback=traceback.format_exc(), failed_at=_now()))
    return True


def _run_unit(unit: dict, st: dict, clip_runner, campaign_root: Path, max_attempts: int) -> None:
    """One GPU clip in the subprocess: retry a CUDA OOM once with state offload forced,
    and record a death the child could not record itself (crash, or a signal: walltime,
    preemption, scancel). A signal death, or any death while this shard is draining, is
    an interruption: the clip stays pending and is retried after the requeue."""
    d = campaign_dirs(campaign_root)
    vid = unit["vid"]
    prior = st["error"]
    attempt = st["attempts"] + 1
    force_offload = bool(prior.get("cuda_oom"))
    print(f"[clip] {vid} tids={unit['n_tids']} frames={unit['n_frames']} "
          f"attempt={attempt}{' offload-forced' if force_offload else ''}", flush=True)
    t0 = time.monotonic()
    rc = clip_runner(unit, attempt, force_offload)
    if rc == EXIT_OOM and attempt < max_attempts and not STOP_REQUESTED:
        print(f"[oom] {vid}: retrying once with state offload forced on", flush=True)
        attempt += 1
        rc = clip_runner(unit, attempt, True)
    interrupted = False
    if rc != 0:
        err_now = _error(d["errors"], vid)
        if int(err_now.get("attempt", 0) or 0) < attempt:      # no error written for THIS attempt
            interrupted = rc < 0 or STOP_REQUESTED
            write_json_atomic(d["errors"] / f"{vid}.json", dict(
                vid=vid, attempt=attempt, error_type="SubprocessExit",
                message=f"exit code {rc}" + (" (killed by a signal)" if rc < 0 else ""),
                # keep the offload flag of an OOM attempt that was then killed
                cuda_oom=(rc == EXIT_OOM) or bool(err_now.get("cuda_oom")) or bool(prior.get("cuda_oom")),
                interrupted=interrupted, failed_at=_now()))
    done = (d["records"] / f"{vid}.jsonl").is_file()
    print(f"[{'committed' if done else 'interrupted' if interrupted else 'failed'}] "
          f"{vid} rc={rc} {time.monotonic() - t0:.0f}s", flush=True)


def process(worklist_path: Path, shard_index: int, shard_count: int, campaign_root: Path,
            checkpoint: Path, anchor_policy: str = "human", max_anchors: int = 16,
            roots: Roots | None = None, clip_runner=None, max_attempts: int = MAX_ATTEMPTS,
            python: str | None = None, poll_seconds: float = 10.0,
            max_wait_seconds: float | None = None, hq_fallback: str = "least-flagged",
            roots_from_env: bool = False, max_gap: int = 60, gap_fill: str = "human",
            keep_span_edges: bool = False, contained_negatives: bool = False,
            direction: str = "both", agree_iou: float = 0.3, speck_floor: int = 20,
            speck_ratio: float = 0.1, dispute_score: float | None = 0.907,
            dispute_rule: str = "higher_score", dispute_winner: str = "strong") -> int:
    """Shard worker: walk the worklist from this shard's offset, claim each pending clip
    and run it. Returns 0 when nothing is left for this shard, 99 when drained by a signal
    (Slurm requeue), 1 when a clip failed terminally. While other shards hold the remaining
    claims this shard waits (`poll_seconds` between passes) so a dead shard's clips are
    picked up once their claims go stale; `max_wait_seconds` bounds that wait (None = until
    the walltime). `roots_from_env`: use `roots` (or the environment) instead of the paths
    frozen in the worklist, for a worklist built on a host with other mounts. `max_gap`,
    `gap_fill`, `keep_span_edges` and `contained_negatives` are handed to every clip
    (run_clip) and recorded in each runs/<vid>.json, as are `direction` and the merge
    thresholds of a `both` run."""
    settings = run_settings(anchor_policy, max_anchors, hq_fallback, max_gap, gap_fill,
                            keep_span_edges, contained_negatives, direction, agree_iou,
                            speck_floor, speck_ratio, dispute_score, dispute_rule, dispute_winner)
    wl = load_worklist(worklist_path)
    wl_roots = Roots.from_dict(wl["roots"])
    units = wl["units"]
    if roots_from_env:
        roots = roots or Roots.from_env()
        units = [rebase_unit_paths(u, wl_roots, roots) for u in units]
        n_missing = sum(1 for u in units if not u.get("video_path"))
        print(f"[worker] roots from the environment, not the worklist: annotations under "
              f"{roots.vidor_ann_root}, videos under {roots.vidor_video_root}; "
              f"{n_missing} unit(s) without a video there", flush=True)
    else:
        roots = roots or wl_roots
    d = campaign_dirs(campaign_root)
    for path in d.values():
        path.mkdir(parents=True, exist_ok=True)
    if clip_runner is None:
        clip_runner = subprocess_runner(worklist_path, campaign_root, checkpoint, anchor_policy,
                                        max_anchors, python, hq_fallback,
                                        roots if roots_from_env else None, max_gap, gap_fill,
                                        keep_span_edges, contained_negatives, direction,
                                        agree_iou, speck_floor, speck_ratio, dispute_score,
                                        dispute_rule, dispute_winner)
    order = shard_order(units, shard_index, shard_count)
    job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID", str(shard_index))
    owner = f"slurm:{job}:{task}" if job else f"local:{socket.gethostname()}:{os.getpid()}:{shard_index}"
    base_prov = base_provenance()          # pinned checkpoint hash; no GPU touched
    print(f"[worker] shard {shard_index}/{shard_count} owner={owner} units={len(order)} "
          + " ".join(f"{k}={v}" for k, v in settings.items()), flush=True)
    failed_here, waited = 0, 0.0
    while not STOP_REQUESTED:
        pending, claimed_any = 0, False
        for unit in order:
            if STOP_REQUESTED:
                break
            if unit_status(unit, campaign_root, max_attempts)["status"] != "pending":
                continue
            pending += 1
            with claim(d["claims"] / f"{unit['vid']}.claim", owner) as owned:
                if not owned:
                    continue
                st = unit_status(unit, campaign_root, max_attempts)
                if st["status"] != "pending":
                    continue
                claimed_any = True
                if not _refuse_on_cpu(unit, roots, campaign_root, anchor_policy, max_anchors,
                                      max_attempts, base_prov, settings):
                    _run_unit(unit, st, clip_runner, campaign_root, max_attempts)
                if unit_status(unit, campaign_root, max_attempts)["status"] == "failed":
                    failed_here += 1
        if STOP_REQUESTED or pending == 0:
            break
        if not claimed_any:
            # Everything left is claimed by other shards: wait rather than leave a tail
            # behind should one of them die.
            if max_wait_seconds is not None and waited >= max_wait_seconds:
                print(f"[worker] {pending} unit(s) still claimed by other shards; leaving them",
                      flush=True)
                break
            time.sleep(poll_seconds)
            waited += poll_seconds
    counts = Counter(r["status"] for r in ledger(wl, campaign_root, max_attempts))
    print(f"[worker] shard {shard_index}: done={counts['done']} failed={counts['failed']} "
          f"pending={counts['pending']}", flush=True)
    return EXIT_DRAINED if STOP_REQUESTED else (1 if failed_here else 0)


def campaign_status(wl: dict, campaign_root: Path, max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Progress of a campaign from the ledger and runs/ alone (no parquet written):
    clips by status, masks and refusals so far, refusals by reason code, GPU wall-hours,
    the failed vids with their error type, and `settings`: the distinct run settings
    (SETTINGS_KEYS) found in runs/*.json, i.e. how the campaign was run."""
    rows = ledger(wl, campaign_root, max_attempts)
    d = campaign_dirs(campaign_root)
    by_reason: Counter = Counter()
    masks = refusals = 0
    gpu_seconds = 0.0
    settings: list[dict] = []
    for r in rows:
        if r["status"] != "done":
            continue
        masks += int(r["masks"] or 0)
        refusals += int(r["refusals"] or 0)
        gpu_seconds += float(r["wall_s"] or 0)
        rp = d["runs"] / f"{r['vid']}.json"
        if rp.is_file():
            try:
                run = json.loads(rp.read_text())
            except (OSError, ValueError):
                continue
            by_reason.update(run.get("refusals") or {})
            s = {k: run[k] for k in SETTINGS_KEYS if k in run}
            if s and s not in settings:
                settings.append(s)
    return dict(
        clips=dict(Counter(r["status"] for r in rows)), units=len(rows),
        masks=masks, refusals=refusals, refusals_by_reason=dict(sorted(by_reason.items())),
        gpu_wall_hours=round(gpu_seconds / 3600, 3), settings=settings,
        interrupted_pending=[r["vid"] for r in rows if r["status"] == "pending" and r["attempts"]],
        failed=[dict(vid=r["vid"], attempts=r["attempts"], error_type=r["error_type"])
                for r in rows if r["status"] == "failed"],
    )


def ledger(wl: dict, campaign_root: Path, max_attempts: int = MAX_ATTEMPTS) -> list[dict]:
    rows = []
    for u in wl["units"]:
        st = unit_status(u, campaign_root, max_attempts)
        run, err = st.get("run", {}), st.get("error", {})
        rows.append(dict(vid=u["vid"], vidstg_split=u["vidstg_split"], n_tids=u["n_tids"],
                         n_frames=u["n_frames"], status=st["status"], attempts=st["attempts"],
                         masks=run.get("masks", ""), refusals=sum((run.get("refusals") or {}).values())
                         if run else "", clip_reason=run.get("reason", ""),
                         wall_s=round(run["wall_s"], 1) if run.get("wall_s") is not None else "",
                         vram_gb=round(run["vram_gb"], 2) if run.get("vram_gb") is not None else "",
                         error_type=err.get("error_type", ""), video_path=u.get("video_path") or ""))
    return rows
