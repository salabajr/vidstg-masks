"""vidstg-masks command line: doctor, build-worklist, plan, process, process-one, status, export,
export-concor, render, transcode."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import CHECKPOINT_SHA256, SAM3_COMMIT, __version__
from .anchors import ANCHOR_POLICIES
from .datasets import Roots
from .merge import DISPUTE_RULES, DISPUTE_WINNERS


def _add_roots(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("roots (override the environment variables of the same name)")
    for name in ("vidstg-root", "vidor-ann-root", "vidor-video-root", "vidor-transcoded-root"):
        g.add_argument(f"--{name}", type=Path)


def _add_campaign(p: argparse.ArgumentParser) -> None:
    p.add_argument("--worklist", type=Path, required=True)
    p.add_argument("--campaign-root", type=Path, required=True)


def _add_policy(p: argparse.ArgumentParser) -> None:
    p.add_argument("--direction", choices=("forward", "backward", "both"), default="both",
                   help="both (default): one SAM pass from the span start and one from the span end, "
                        "merged frame by frame by the pixels rule (docs/PIPELINE.md, Backward pass). "
                        "forward: the pass from the span start alone, half the GPU time. backward: "
                        "the pass from the span end alone")
    p.add_argument("--agree-iou", type=float, default=0.3,
                   help="both: overlap at or above which the two masks of an object mostly match "
                        "(the forward one is preferred); below it the frame is a dispute and refused")
    p.add_argument("--speck-floor", type=int, default=20,
                   help="both: a mask under this many pixels is no mask")
    p.add_argument("--speck-ratio", type=float, default=0.1,
                   help="both: with two real masks, the one under this share of the other is no mask")
    p.add_argument("--dispute-rule", choices=DISPUTE_RULES, default="higher_score",
                   help="both: what to do on a disputed frame (two real masks that do not overlap). "
                        "refuse: no mask. higher_score (default): the pass with the higher presence score "
                        "(mask_confidence) when that score is at least --dispute-score, else no mask (equal "
                        "scores: forward). forward_score: the forward mask when its score is at least "
                        "--dispute-score, else no mask. The defaults are the measured setting, higher_score 0.907 "
                        "with --dispute-winner strong (docs/PIPELINE.md, Backward pass)")
    p.add_argument("--dispute-score", type=float, default=None,
                   help="both: the presence-score threshold of --dispute-rule higher_score / forward_score "
                        "(default 0.907 with them, not allowed with refuse)")
    p.add_argument("--dispute-winner", choices=DISPUTE_WINNERS, default="strong",
                   help="both: how a tie-break mask meets its neighbours. weak: it yields the "
                        "pixels it shares with a neighbour whose pass was the only candidate (and a backward "
                        "win also yields to a neighbour's forward mask); left under --speck-floor it is "
                        "refused. strong (default): it takes the pixels an agreed neighbour shares with it and is "
                        "refused together with a neighbour that is the only candidate of the other pass")
    p.add_argument("--anchor-policy", choices=ANCHOR_POLICIES, default="human_gap",
                   help="human_gap (default): the VidOR box at every human keyframe, capped by "
                        "--max-anchors, plus the coverage rule (--max-gap / --gap-fill), no "
                        "quality gate. human: the keyframes alone, without the coverage rule "
                        "(the earlier default). hq: the keyframes filtered by the quality "
                        "gate, then the coverage rule")
    p.add_argument("--max-anchors", type=int, default=16)
    p.add_argument("--hq-fallback", choices=("least-flagged", "refuse"), default="least-flagged",
                   help="hq only: an object with no clean keyframe keeps its least-flagged "
                        "keyframes (recorded in provenance) or is refused (no_hq_anchor)")
    p.add_argument("--max-gap", type=int, default=60,
                   help="coverage rule (human_gap and hq; not applied under human): no stretch "
                        "of an object's span longer than this many frames without an anchor; "
                        "each gap is filled with the human keyframe nearest its middle (the "
                        "least-flagged one under hq); 0 disables the gap rule")
    p.add_argument("--gap-fill", choices=("human", "any"), default="human",
                   help="what may fill a gap: a human keyframe only (default), or with 'any' "
                        "the tracker box nearest the middle where no human keyframe lies in it")
    p.add_argument("--keep-span-edges", action="store_true",
                   help="hq only: always anchor at the object's first and last human keyframe "
                        "in its span, whatever the gate says")
    p.add_argument("--contained-negatives", action=argparse.BooleanOptionalAction, default=True,
                   help="on by default, any policy: overlap-aware prompts from the boxes alone - "
                        "a negative click for the container at each contained small object's "
                        "centre, and the container co-prompted at the contained object's "
                        "anchors (recorded in prompt_payload.contained_negatives); "
                        "--no-contained-negatives gives the plain box prompts")


def _add_vids(p: argparse.ArgumentParser) -> None:
    p.add_argument("--vids", nargs="+", help="restrict to these video ids")
    p.add_argument("--vids-file", type=Path, help="one vid per line (first token; # comments)")


def _add_runner(p: argparse.ArgumentParser) -> None:
    """Shared by process and process-one."""
    _add_roots(p)
    _add_campaign(p)
    _add_policy(p)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--roots-from-env", action="store_true",
                   help="take the dataset roots from the environment / --*-root flags and rebase "
                        "every unit's paths onto them, instead of the roots frozen in the worklist "
                        "(worklist built on a host with other mount points, or data moved since)")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vidstg-masks",
                                 description="SAM 3.1 mask tracks for every VidSTG relation (subject and object)")
    ap.add_argument("--version", action="version", version=f"vidstg-masks {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="check roots, tools, sam3, checkpoint hash and the known dataset counts")
    _add_roots(d)
    d.add_argument("--checkpoint", type=Path, default=os.environ.get("SAM31_CHECKPOINT") or None)
    d.add_argument("--sam3-repo", type=Path, default=os.environ.get("SAM31_REPO_ROOT") or None)
    d.add_argument("--skip-hash", action="store_true", help="skip the sha256 of the 3.5 GB checkpoint")
    d.add_argument("--skip-facts", action="store_true", help="skip loading the annotations")
    d.add_argument("--skip-gpu-libs", action="store_true", help="do not import torch / sam3")
    d.add_argument("--vid", help="also probe one video: resolve, decode count vs annotation, black check")

    b = sub.add_parser("build-worklist", help="select videos and write <campaign>/worklist.json")
    _add_roots(b)
    _add_vids(b)
    b.add_argument("--split", choices=("train", "val", "test", "all"), required=True,
                   help="VidSTG annotation file(s) to draw videos from")
    b.add_argument("--campaign-root", type=Path, required=True)
    b.add_argument("--limit", type=int)
    b.add_argument("--if-missing", action="store_true",
                   help="do nothing when <campaign>/worklist.json already exists (resume)")
    b.add_argument("--no-assert-counts", action="store_true",
                   help="do not require the release counts (VidSTG records per split, 7,835 VidOR "
                        "annotation files); for a deliberately partial copy of the dataset")

    pr = sub.add_parser("process", help="run one shard of the worklist (one GPU)")
    _add_runner(pr)
    pr.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    pr.add_argument("--shard-count", type=int, default=1)
    pr.add_argument("--max-attempts", type=int, default=2,
                    help="attempts for a CUDA out-of-memory clip (the retry forces state offload); "
                         "an interrupted clip (killed by a signal) is retried up to 5 times, any "
                         "other error is terminal after one attempt")
    pr.add_argument("--python", help="interpreter for the per-clip subprocess (default: this one)")

    po = sub.add_parser("process-one", help="(internal) one clip in this process")
    _add_runner(po)
    po.add_argument("--vid", required=True)
    po.add_argument("--attempt", type=int, default=1)
    po.add_argument("--force-offload", action="store_true")
    po.add_argument("--checkpoint-hash", help="sha256 already verified by the parent shard")

    pl = sub.add_parser("plan", help="print the CPU plan for one or more vids of a worklist (no GPU)")
    pl.add_argument("--worklist", type=Path, required=True)
    pl.add_argument("--vids", nargs="+", required=True)
    _add_policy(pl)

    st = sub.add_parser("status", help="progress of a campaign: clips done/failed/pending, masks, "
                                       "refusals by reason, GPU hours, failed vids (writes nothing)")
    _add_campaign(st)
    st.add_argument("--json", action="store_true", help="machine-readable output")

    ex = sub.add_parser("export", help="records/*.jsonl -> export/ parquet tables, ledger, manifest")
    _add_campaign(ex)
    ex.add_argument("--output-dir", type=Path)
    ex.add_argument("--no-integrity", action="store_true")

    ec = sub.add_parser("export-concor", help="records/*.jsonl -> export/concor/: one ConCor Video "
                                              "record per relation (schema concor-video-tracklet-bcc-v2), "
                                              "their samples / tracklets / links / verification tables")
    _add_campaign(ec)
    ec.add_argument("--output-dir", type=Path, help="default <campaign>/export/concor")
    ec.add_argument("--captions", type=Path, help="captions.jsonl: {vid, subject_tid, predicate, object_tid, "
                                                 "text, spans: {\"<tid>\": [[start, end], ...]}}; relations "
                                                 "without a caption go to tracklets.parquet only")

    r = sub.add_parser("render", help="QA overlay mp4 for one clip")
    _add_roots(r)
    r.add_argument("--vid", required=True)
    r.add_argument("--campaign-root", type=Path, required=True)
    r.add_argument("--out", type=Path, help="default <campaign>/overlays/<vid>.mp4")
    r.add_argument("--alpha", type=float, default=0.45, help="mask tint strength, 0 to 1")
    r.add_argument("--no-labels", action="store_true",
                   help="do not write `<tid>:<category>` at each box (and \"(no mask)\" where the object has a box but no mask)")
    r.add_argument("--side-by-side", action="store_true",
                   help="the untouched frame on the left and the painted one on the right (twice the width)")
    r.add_argument("--crf", type=int, default=20, help="libx264 quality, lower is larger (default 20)")

    t = sub.add_parser("transcode", help="H.264 re-encode into VIDOR_TRANSCODED_ROOT for videos that decode black; "
                                         "with --campaign-root the repaired clips are queued again for `process`")
    _add_roots(t)
    _add_vids(t)
    t.add_argument("--worklist", type=Path, help="with --campaign-root: transcode every unit the campaign refused as "
                                                 "decode_black or frame_size_mismatch")
    t.add_argument("--campaign-root", type=Path,
                   help="a refused unit whose re-encode decodes with the annotation's frame count and size has its "
                        "refusal records moved to <campaign>/superseded/ so the next `process` runs it")
    t.add_argument("--all-black", action="store_true",
                   help="with --worklist: probe every unit's video and transcode the black ones")
    return ap


# ── doctor ──────────────────────────────────────────────────────────────────

def _line(ok: bool | None, label: str, detail: str = "") -> None:
    tag = "OK  " if ok else ("FAIL" if ok is False else "--  ")
    print(f"[{tag}] {label}" + (f": {detail}" if detail else ""))


def _doctor_gpu_libs() -> None:
    try:
        import torch
        cuda = torch.cuda.is_available()
        _line(True, "torch", f"{torch.__version__} · cuda "
              f"{'available: ' + torch.cuda.get_device_name(0) if cuda else 'NOT available (fine on a login node)'}")
    except Exception as e:  # noqa: BLE001
        _line(None, "torch", f"not importable here ({type(e).__name__}); needed on GPU nodes only")
    try:
        import sam3  # noqa: F401
        _line(True, "sam3", str(Path(sam3.__file__).parent))
    except Exception as e:  # noqa: BLE001
        _line(None, "sam3", f"not importable here ({type(e).__name__}); needed on GPU nodes only")


def cmd_doctor(args) -> int:
    from .datasets import build_vidor_index, known_facts_check, load_vidstg
    from .video import have_ffmpeg

    failures = 0
    print(f"vidstg-masks {__version__} · python {sys.version.split()[0]} · {sys.executable}")
    try:
        roots = Roots.from_args(args)
    except KeyError as e:
        _line(False, "roots", str(e))
        return 1
    for name in ("vidstg_root", "vidor_ann_root", "vidor_video_root", "vidor_transcoded_root"):
        p = getattr(roots, name)
        if p is None:
            _line(None, name, "not set (optional)")
        else:
            ok = p.is_dir()
            failures += not ok
            _line(ok, name, f"{p}{'' if ok else ' (missing)'}")
    for s in ("train", "val", "test"):
        p = roots.vidstg_file(s)
        _line(p.is_file(), f"vidstg {s}", str(p))
        failures += not p.is_file()

    for tool, path in have_ffmpeg().items():
        _line(bool(path), tool, path or "not on PATH")
        failures += not path
    for mod in ("numpy", "cv2", "pycocotools", "pyarrow"):
        try:
            m = __import__(mod)
            _line(True, mod, getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            _line(False, mod, str(e))
            failures += 1
    if args.skip_gpu_libs:
        _line(None, "torch / sam3", "import skipped")
    else:
        _doctor_gpu_libs()
    if args.sam3_repo:

        try:
            head = subprocess.run(["git", "-C", str(args.sam3_repo), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
            ok = head == SAM3_COMMIT
            failures += not ok
            _line(ok, "sam3 commit", f"{head[:12]} {'== pinned' if ok else '!= pinned ' + SAM3_COMMIT[:12]}")
        except Exception as e:  # noqa: BLE001
            _line(False, "sam3 commit", f"{args.sam3_repo}: {e}")
            failures += 1
    if args.checkpoint:
        cp = Path(args.checkpoint)
        if not cp.is_file():
            _line(False, "checkpoint", f"{cp} missing (scripts/download_model.sh)")
            failures += 1
        elif args.skip_hash:
            _line(None, "checkpoint", f"{cp} ({cp.stat().st_size / 1e9:.2f} GB) hash skipped")
        else:
            from .sam_session import checkpoint_sha256
            h = checkpoint_sha256(cp)
            ok = h == CHECKPOINT_SHA256
            failures += not ok
            _line(ok, "checkpoint sha256", f"{h[:16]}… {'== pinned' if ok else '!= pinned ' + CHECKPOINT_SHA256[:16]}")
    else:
        _line(None, "checkpoint", "not given (--checkpoint or SAM31_CHECKPOINT)")

    if not args.skip_facts and all(roots.vidstg_file(s).is_file() for s in ("train", "val", "test")) \
            and roots.vidor_ann_root.is_dir():
        records = load_vidstg(roots)
        index = build_vidor_index(roots)
        facts = known_facts_check(records, index)
        for k, v in facts["counts"].items():
            _line(facts["matches"][k], f"known fact {k}", f"{v:,} (expected {facts['expected'][k]:,})")
        _line(facts["videos_with_vidor_annotation"] == facts["counts"]["vidstg_videos"],
              "VidSTG videos with a VidOR annotation",
              f"{facts['videos_with_vidor_annotation']:,} / {facts['counts']['vidstg_videos']:,}")
        print(f"       records per split: {facts['records_per_split']}")
        failures += not facts["ok"]
        if args.vid:
            from .video import decode_check
            from .datasets import load_vidor, resolve_video
            if args.vid not in index:
                _line(False, f"vid {args.vid}", "no VidOR annotation")
                failures += 1
            else:
                ann = load_vidor(index[args.vid])
                video = resolve_video(roots, ann)
                _line(video is not None, f"vid {args.vid} video", str(video or ann["video_path"] + " missing"))
                if video is not None:
                    chk = decode_check(video, ann["frame_count"], (ann["width"], ann["height"]))
                    _line(chk["frame_count_ok"], "decoded frame count",
                          f"{chk['frame_count']} vs annotation {ann['frame_count']}")
                    _line(chk["size_ok"], "decoded size", f"{chk['size']} vs annotation {(ann['width'], ann['height'])}")
                    _line(not chk["black"], "black-decode probe",
                          "black frames (VP6F class; run `vidstg-masks transcode`)" if chk["black"] else "frames have content")
                    failures += (not chk["frame_count_ok"]) + (not chk["size_ok"]) + chk["black"]
    elif not args.skip_facts:
        _line(None, "known facts", "skipped: annotation set incomplete")
    print(f"{'all checks passed' if not failures else str(failures) + ' check(s) failed'}")
    return 1 if failures else 0


# ── other commands ──────────────────────────────────────────────────────────

def cmd_build_worklist(args) -> int:
    from .worker import build_worklist, read_vids_file, save_worklist

    path = args.campaign_root.resolve() / "worklist.json"
    if args.if_missing and path.is_file():
        print(f"worklist exists, kept: {path}")
        return 0
    roots = Roots.from_args(args)
    vids = None
    if args.vids or args.vids_file:
        vids = list(args.vids or []) + (read_vids_file(args.vids_file) if args.vids_file else [])
    wl = build_worklist(roots, args.split, vids, args.limit, assert_counts=not args.no_assert_counts)
    save_worklist(path, wl)
    print(json.dumps({"worklist": str(path), "counts": wl["counts"]}, indent=1))
    return 0


def _check_dispute(args) -> None:
    from .merge import DEFAULT_DISPUTE_SCORE, check_dispute_settings

    try:
        if args.dispute_rule != "refuse" and args.dispute_score is None:
            args.dispute_score = DEFAULT_DISPUTE_SCORE       # a score rule without a threshold takes the measured one
        check_dispute_settings(args.dispute_rule, args.dispute_score, args.dispute_winner)
    except ValueError as e:
        raise SystemExit(f"--dispute-rule / --dispute-score / --dispute-winner: {e}")


def cmd_process(args) -> int:
    from .worker import install_signal_handlers, process

    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must satisfy 0 <= index < shard-count")
    _check_dispute(args)
    install_signal_handlers()
    roots = Roots.from_args(args) if args.roots_from_env else None
    return process(args.worklist.resolve(), args.shard_index, args.shard_count,
                   args.campaign_root.resolve(), args.checkpoint.resolve(), args.anchor_policy,
                   args.max_anchors, roots=roots, max_attempts=args.max_attempts,
                   python=args.python, hq_fallback=args.hq_fallback,
                   roots_from_env=args.roots_from_env, max_gap=args.max_gap,
                   gap_fill=args.gap_fill, keep_span_edges=args.keep_span_edges,
                   contained_negatives=args.contained_negatives, direction=args.direction,
                   agree_iou=args.agree_iou, speck_floor=args.speck_floor,
                   speck_ratio=args.speck_ratio, dispute_score=args.dispute_score,
                   dispute_rule=args.dispute_rule, dispute_winner=args.dispute_winner)


def cmd_process_one(args) -> int:
    from .worker import process_one_main

    _check_dispute(args)
    roots = Roots.from_args(args) if args.roots_from_env else None
    return process_one_main(args.worklist.resolve(), args.vid, args.campaign_root.resolve(),
                            args.checkpoint.resolve(), args.anchor_policy, args.max_anchors,
                            args.attempt, args.force_offload, args.hq_fallback,
                            checkpoint_hash=args.checkpoint_hash, roots=roots,
                            max_gap=args.max_gap, gap_fill=args.gap_fill,
                            keep_span_edges=args.keep_span_edges,
                            contained_negatives=args.contained_negatives, direction=args.direction,
                            agree_iou=args.agree_iou, speck_floor=args.speck_floor,
                            speck_ratio=args.speck_ratio, dispute_score=args.dispute_score,
                            dispute_rule=args.dispute_rule, dispute_winner=args.dispute_winner)


def cmd_status(args) -> int:
    from .worker import campaign_status, load_worklist

    s = campaign_status(load_worklist(args.worklist), args.campaign_root.resolve())
    if args.json:
        print(json.dumps(s, indent=1))
        return 0
    c = s["clips"]
    print(f"campaign {args.campaign_root.resolve()} · worklist {args.worklist} · {s['units']} clips")
    print(f"  done {c.get('done', 0)} · failed {c.get('failed', 0)} · pending {c.get('pending', 0)}"
          + (f" (of which {len(s['interrupted_pending'])} interrupted, awaiting retry)"
             if s["interrupted_pending"] else ""))
    print(f"  masks {s['masks']:,} · refusals {s['refusals']:,} · GPU wall-hours {s['gpu_wall_hours']}")
    if s["refusals_by_reason"]:
        print("  refusals by reason: " + ", ".join(f"{k} {v:,}" for k, v in s["refusals_by_reason"].items()))
    for st_ in s["settings"]:
        print("  run settings: " + " ".join(f"{k}={v}" for k, v in st_.items()))
    for f in s["failed"]:
        print(f"  failed {f['vid']} attempts={f['attempts']} {f['error_type']}")
    return 0


def cmd_plan(args) -> int:
    from .anchors import add_contained_negatives, describe_plan, prompts_for_plan, summarize_negatives
    from .worker import load_worklist, plan_unit

    wl = load_worklist(args.worklist)
    roots = Roots.from_dict(wl["roots"])
    units = {u["vid"]: u for u in wl["units"]}
    for vid in args.vids:
        if vid not in units:
            print(f"{vid}: not in worklist", file=sys.stderr)
            continue
        plan, video = plan_unit(units[vid], roots, args.max_anchors, args.anchor_policy,
                                args.hq_fallback, max_gap=args.max_gap, gap_fill=args.gap_fill,
                                keep_span_edges=args.keep_span_edges)
        print(describe_plan(plan, video))
        if args.anchor_policy == "hq":
            from .anchor_quality import describe
            for t in plan["tids"]:
                print(f"   hq tid {t}: {describe(plan, t)}")
        if args.contained_negatives:
            prov = add_contained_negatives(plan, prompts_for_plan(plan))
            print(f"   {summarize_negatives(prov)}")
            for t, v in prov.items():
                if v["n_neg_clicks"]:
                    print(f"   tid {t}: {v['n_neg_clicks']} negative clicks at fids "
                          f"{list(v['neg_clicks'])} · co-prompted at {v['co_prompt_fids']} "
                          f"(tracker boxes: {v['co_prompt_tracker_fids']})")
    return 0


def cmd_export(args) -> int:
    from .export import export_campaign

    manifest = export_campaign(args.worklist.resolve(), args.campaign_root.resolve(),
                               args.output_dir.resolve() if args.output_dir else None,
                               check_integrity=not args.no_integrity)
    shown = dict(manifest)
    shown["integrity"] = dict(manifest["integrity"], problems=manifest["integrity"]["problems"][:10])
    print(json.dumps(shown, indent=1))
    return 0 if manifest["integrity"]["ok"] else 1


def cmd_export_concor(args) -> int:
    from .concor import export_concor

    manifest = export_concor(args.worklist.resolve(), args.campaign_root.resolve(),
                             args.output_dir.resolve() if args.output_dir else None,
                             args.captions.resolve() if args.captions else None)
    print(json.dumps(dict(manifest, problems=manifest["problems"][:10]), indent=1))
    return 0 if manifest["ok"] else 1


def cmd_render(args) -> int:
    from .render import render_overlay
    from .worker import load_worklist

    roots = Roots.from_args(args)
    records = args.campaign_root / "records" / f"{args.vid}.jsonl"
    if not records.is_file():
        raise SystemExit(f"no records for {args.vid} at {records}")
    relations = None
    wl_path = args.campaign_root / "worklist.json"
    if wl_path.is_file():
        unit = next((u for u in load_worklist(wl_path)["units"] if u["vid"] == args.vid), None)
        relations = unit["relations"] if unit else None
    out = args.out or (args.campaign_root / "overlays" / f"{args.vid}.mp4")
    n = render_overlay(args.vid, records, roots, out, relations, args.alpha, crf=args.crf,
                       labels=not args.no_labels, side_by_side=args.side_by_side)
    print(f"{n} frames -> {out}")
    return 0


def cmd_transcode(args) -> int:
    from .datasets import build_vidor_index, load_vidor
    from .video import is_black, probe_frame_count, probe_size, transcode_h264
    from .worker import load_worklist, read_vids_file

    roots = Roots.from_args(args)
    if roots.vidor_transcoded_root is None:
        raise SystemExit("set VIDOR_TRANSCODED_ROOT (or --vidor-transcoded-root) for the output")
    vids = list(args.vids or []) + (read_vids_file(args.vids_file) if args.vids_file else [])
    if args.worklist:
        wl = load_worklist(args.worklist)
        for u in wl["units"]:
            if args.all_black:
                if u.get("video_path") and is_black(Path(u["video_path"])):
                    vids.append(u["vid"])
            elif args.campaign_root:
                rp = args.campaign_root / "runs" / f"{u['vid']}.json"
                if rp.is_file() and json.loads(rp.read_text()).get("reason") in REQUEUE_REASONS:
                    vids.append(u["vid"])
    if not vids:
        print("nothing to transcode")
        return 0
    index = build_vidor_index(roots)
    rc = 0
    for vid in dict.fromkeys(vids):
        ann = load_vidor(index[vid])
        src = roots.vidor_video_root / ann["video_path"]
        dst = roots.vidor_transcoded_root / ann["video_path"]
        if not src.is_file():
            print(f"{vid}: source missing {src}")
            rc = 1
            continue
        transcode_h264(src, dst)
        n = probe_frame_count(dst)
        size = probe_size(dst)
        black = is_black(dst)
        ok = n == ann["frame_count"] and size == (ann["width"], ann["height"]) and not black
        print(f"{vid}: {dst} frames {n} vs annotation {ann['frame_count']} · size {size} vs "
              f"{(ann['width'], ann['height'])} · black={black} · {'OK' if ok else 'NOT USABLE'}")
        if not ok:
            rc = 1
        elif args.campaign_root:
            moved = requeue_refused_unit(args.campaign_root, vid, dst)
            if moved:
                print(f"{vid}: refusal records moved to {args.campaign_root / 'superseded'}, worklist video_path -> {dst}; "
                      "the clip is pending again")
    return rc


REQUEUE_REASONS = ("decode_black", "frame_size_mismatch")


def requeue_refused_unit(campaign_root: Path, vid: str, new_video: Path | None = None) -> bool:
    """A clip the campaign refused for a video defect that a transcode repairs (REQUEUE_REASONS):
    move its runs/, records/ and errors/ files under <campaign>/superseded/<kind>/, keeping them for
    the record, so `unit_status` sees the clip as pending; and point the unit's `video_path` in
    <campaign>/worklist.json at `new_video` (the worklist froze the path of the video that failed,
    and the worker reads that path, not the transcoded root). Returns False when the run was not
    such a refusal (nothing is touched)."""
    from .records import write_json_atomic

    rp = campaign_root / "runs" / f"{vid}.json"
    if not rp.is_file() or json.loads(rp.read_text()).get("reason") not in REQUEUE_REASONS:
        return False
    for kind, name in (("runs", f"{vid}.json"), ("records", f"{vid}.jsonl"), ("errors", f"{vid}.json")):
        src = campaign_root / kind / name
        if src.is_file():
            dst = campaign_root / "superseded" / kind / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst)
    wl_path = campaign_root / "worklist.json"
    if new_video is not None and wl_path.is_file():
        wl = json.loads(wl_path.read_text())
        for u in wl["units"]:
            if u["vid"] == vid:
                u["video_path"] = str(Path(new_video).resolve())
        write_json_atomic(wl_path, wl)
    return True


COMMANDS = {"doctor": cmd_doctor, "build-worklist": cmd_build_worklist, "process": cmd_process,
            "process-one": cmd_process_one, "plan": cmd_plan, "status": cmd_status,
            "export": cmd_export, "export-concor": cmd_export_concor, "render": cmd_render,
            "transcode": cmd_transcode}


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    raise SystemExit(COMMANDS[args.command](args))


if __name__ == "__main__":
    main()
