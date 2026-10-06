import json
import os
import threading
import time

import numpy as np
import pytest

from conftest import N_FRAMES, VID_A, VID_B, VID_C, VID_D
from vidstg_masks.anchors import add_contained_negatives, in_span_fids, plan_clip, prompts_for_plan
from vidstg_masks.datasets import load_vidor, load_vidstg, records_by_vid
from vidstg_masks.records import read_jsonl, validate_record
from vidstg_masks.worker import (MAX_OBJECTS, build_worklist, claim, clip_records, ledger,
                                 load_worklist, plan_unit, precheck_clip, process,
                                 read_vids_file, save_worklist, shard_order, unit_records,
                                 unit_status)


# ── worklist ────────────────────────────────────────────────────────────────

def test_build_worklist_all_splits(roots):
    wl = build_worklist(roots, "all")
    vids = [u["vid"] for u in wl["units"]]
    assert vids == sorted([VID_A, VID_B, VID_C, VID_D])
    by = {u["vid"]: u for u in wl["units"]}
    assert by[VID_A]["vidstg_split"] == "train" and by[VID_C]["vidstg_split"] == "test"
    assert by[VID_A]["relations"] == [[0, "hold", 1], [0, "watch", 2]] and by[VID_A]["segment"] == [0, 11]
    assert by[VID_A]["n_tids"] == 3 and by[VID_C]["n_tids"] == 17
    assert by[VID_D]["video_path"] is None and by[VID_A]["video_path"].endswith(f"{VID_A}.mp4")
    assert wl["counts"]["video_missing"] == 2 and wl["counts"]["too_many_objects"] == 1   # C and D lack a video
    assert wl["roots"] == roots.to_dict()


def test_build_worklist_split_vids_limit(roots, tmp_path):
    assert [u["vid"] for u in build_worklist(roots, "val")["units"]] == [VID_B, VID_D]
    assert [u["vid"] for u in build_worklist(roots, "all", vids=[VID_B, "nope"])["units"]] == [VID_B]
    assert len(build_worklist(roots, "all", limit=2)["units"]) == 2
    f = tmp_path / "vids.txt"
    f.write_text(f"# vid  split  relations\n{VID_A}  calib  8  5\n\n{VID_B} dev 1\n")
    assert read_vids_file(f) == [VID_A, VID_B]
    p = tmp_path / "c" / "worklist.json"
    save_worklist(p, build_worklist(roots, "train"))
    assert load_worklist(p)["units"][0]["vid"] == VID_A


def test_unit_records_reproduce_the_plan(roots):
    wl = build_worklist(roots, "all")
    recs = records_by_vid(load_vidstg(roots))
    for u in wl["units"]:
        ann = load_vidor(u["vidor_ann"])
        a = plan_clip(u["vid"], recs[u["vid"]], ann, u["vidstg_split"])
        b = plan_clip(u["vid"], unit_records(u), ann, u["vidstg_split"])
        for k in ("relations", "tids", "seg", "spans", "anchors", "prop_span", "cats"):
            assert a[k] == b[k], (u["vid"], k)


def test_sharding_covers_every_unit_exactly_once():
    units = [{"vid": str(i)} for i in range(10)]
    for count in (1, 2, 3, 4, 10, 12):
        block = -(-len(units) // count)
        for i in range(count):
            order = shard_order(units, i, count)
            assert sorted(u["vid"] for u in order) == sorted(u["vid"] for u in units)
            assert order[0] == units[(i * block) % len(units)]       # starts at its own block
    with pytest.raises(ValueError):
        shard_order(units, 3, 3)


# ── claims ──────────────────────────────────────────────────────────────────

def test_claims_are_exclusive(tmp_path):
    p = tmp_path / "claims" / "v.claim"
    with claim(p, "a", heartbeat=1) as got_a:
        assert got_a and p.is_file()
        with claim(p, "b", heartbeat=1) as got_b:
            assert not got_b
        with claim(p, "a", heartbeat=1) as again:   # same owner re-enters
            assert again
    assert not p.exists()


def test_stale_claim_is_recovered(tmp_path):
    p = tmp_path / "claims" / "v.claim"
    p.parent.mkdir()
    p.write_text(json.dumps({"owner": "dead", "pid": 1}))
    old = time.time() - 31 * 60
    os.utime(p, (old, old))
    with claim(p, "live", heartbeat=1) as got:
        assert got
        assert json.loads(p.read_text())["owner"] == "live"
    assert list(p.parent.glob("v.claim.stale-*"))


def test_concurrent_claim_has_one_owner(tmp_path):
    p = tmp_path / "v.claim"
    results, barrier, release = [], threading.Barrier(2), threading.Event()

    def contender(owner):
        barrier.wait()
        with claim(p, owner, heartbeat=1) as got:
            results.append(got)
            if got:
                release.wait(3)
    ts = [threading.Thread(target=contender, args=(f"w{i}",)) for i in range(2)]
    for t in ts:
        t.start()
    while len(results) < 2:
        time.sleep(0.01)
    release.set()
    for t in ts:
        t.join(5)
    assert sorted(results) == [False, True]


# ── per-clip CPU path ───────────────────────────────────────────────────────

def test_frame_count_mismatch_refuses_the_clip(roots, data, base_prov):
    wl = build_worklist(roots, "train")
    u = wl["units"][0]
    plan, video = plan_unit(u, roots, 16, "human")
    assert precheck_clip(plan, video)[0] is None
    plan["frame_count"] = N_FRAMES + 1                    # annotation disagrees with the decode
    reason, details = precheck_clip(plan, video)
    assert reason == "frame_count_mismatch" and details["frame_count"] == N_FRAMES
    recs, refusals = clip_records(plan, None, base_prov, clip_reason=reason)
    assert set(refusals) == {"frame_count_mismatch"}
    assert len(recs) == sum(len(in_span_fids(plan, t)) for t in plan["tids"]) == 12 + 10 + 12
    for r in recs:
        validate_record(r)
        assert r["rle"] is None and r["reason_code"] == "frame_count_mismatch"
    # other clip-level defects
    plan["frame_count"] = N_FRAMES
    assert precheck_clip(plan, None)[0] == "video_missing"
    plan["tids"] = list(range(MAX_OBJECTS + 1))
    assert precheck_clip(plan, video)[0] == "too_many_objects"


def test_clip_records_masks_and_refusals(roots, base_prov):
    wl = build_worklist(roots, "train")
    plan, _ = plan_unit(wl["units"][0], roots, 16, "human")
    m = np.zeros((48, 64), bool)
    m[10:30, 10:30] = True
    per_frame = {f: {0: (m, 0.9), 1: (m if f != 5 else np.zeros_like(m), 0.8)} for f in range(12)}
    del per_frame[11]                                     # propagation did not reach frame 11
    recs, refusals = clip_records(plan, per_frame, base_prov)
    by = {(r["tid"], r["fid"]): r for r in recs}
    assert by[(0, 3)]["rle"]["size"] == [48, 64] and by[(0, 3)]["mask_confidence"] == 0.9
    assert by[(1, 5)]["reason_code"] == "empty_mask"      # empty mask -> refusal
    assert by[(0, 11)]["reason_code"] == "not_tracked"    # the pass never reached the frame -> refusal
    assert all(by[(2, f)]["reason_code"] == "no_human_keyframe" for f in range(12))
    assert refusals == {"empty_mask": 1, "not_tracked": 2, "no_human_keyframe": 12}
    assert by[(0, 3)]["prompt_payload"] == {"anchor_fids": [0, 4, 8], "ref_anchor_fid": 0, "anchor_policy": "human"}
    assert by[(0, 3)]["prompt_mode"] == "pvs_box_multianchor" and by[(0, 3)]["split"] == "train"
    assert by[(0, 4)]["box_generated"] == 0 and by[(0, 5)]["box_tracker"] == "kcf"
    for r in recs:
        validate_record(r)


# ── the shard worker with an injected clip runner ───────────────────────────

def fake_runner(roots, campaign, calls):
    """Stands in for the GPU subprocess: writes full-mask records for the clip."""
    from vidstg_masks.records import base_provenance, write_jsonl_atomic, write_json_atomic

    def run(unit, attempt, force_offload):
        calls.append((unit["vid"], attempt, force_offload))
        plan, _ = plan_unit(unit, roots, 16, "human")
        m = np.ones((48, 64), bool)
        per_frame = {f: {t: (m, 1.0) for t in plan["tids"]} for f in range(12)}
        recs, refusals = clip_records(plan, per_frame, base_provenance(code_commit="fake"))
        write_jsonl_atomic(campaign / "records" / f"{unit['vid']}.jsonl", recs)
        write_json_atomic(campaign / "runs" / f"{unit['vid']}.json",
                          dict(vid=unit["vid"], status="done", masks=len(recs), refusals=dict(refusals),
                               wall_s=1.0, vram_gb=2.0))
        return 0
    return run


@pytest.fixture
def campaign(roots, tmp_path):
    c = tmp_path / "campaign"
    save_worklist(c / "worklist.json", build_worklist(roots, "all"))
    return c


def test_process_runs_gpu_clips_refuses_cpu_clips_and_resumes(roots, campaign):
    calls = []
    rc = process(campaign / "worklist.json", 0, 1, campaign, campaign / "ckpt.pt", roots=roots,
                 clip_runner=fake_runner(roots, campaign, calls))
    assert rc == 0
    assert sorted(v for v, _, _ in calls) == [VID_A, VID_B]           # only clips that need SAM
    rows = {r["vid"]: r for r in ledger(load_worklist(campaign / "worklist.json"), campaign)}
    assert all(r["status"] == "done" for r in rows.values())
    c = list(read_jsonl(campaign / "records" / f"{VID_C}.jsonl"))
    assert c and all(r["reason_code"] == "too_many_objects" and r["rle"] is None for r in c)
    assert len(c) == 17 * 12
    d = list(read_jsonl(campaign / "records" / f"{VID_D}.jsonl"))
    assert d and all(r["reason_code"] == "video_missing" for r in d)
    assert json.loads((campaign / "runs" / f"{VID_C}.json").read_text())["reason"] == "too_many_objects"
    assert not list((campaign / "claims").glob("*.claim"))
    # second pass: everything is done, the runner is never called
    calls.clear()
    assert process(campaign / "worklist.json", 0, 1, campaign, campaign / "ckpt.pt", roots=roots,
                   clip_runner=fake_runner(roots, campaign, calls)) == 0
    assert calls == []


def test_process_skips_claimed_and_failed_units(roots, campaign):
    from vidstg_masks.worker import EXIT_OOM
    (campaign / "claims").mkdir()
    (campaign / "claims" / f"{VID_A}.claim").write_text(json.dumps({"owner": "other"}))
    attempts = []

    def oom_then_ok(unit, attempt, force_offload):
        attempts.append((unit["vid"], attempt, force_offload))
        if unit["vid"] == VID_B and attempt == 1:
            from vidstg_masks.records import write_json_atomic
            write_json_atomic(campaign / "errors" / f"{VID_B}.json",
                              dict(vid=VID_B, attempt=attempt, error_type="OutOfMemoryError",
                                   message="CUDA out of memory", cuda_oom=True))
            return EXIT_OOM
        return fake_runner(roots, campaign, [])(unit, attempt, force_offload)
    rc = process(campaign / "worklist.json", 0, 1, campaign, campaign / "ckpt.pt", roots=roots,
                 clip_runner=oom_then_ok, poll_seconds=0.01, max_wait_seconds=0.05)
    # A is held by another worker (fresh claim) -> left pending; B OOMs once and is retried with offload
    assert attempts == [(VID_B, 1, False), (VID_B, 2, True)]
    assert rc == 0
    st = {u["vid"]: unit_status(u, campaign) for u in load_worklist(campaign / "worklist.json")["units"]}
    assert st[VID_A]["status"] == "pending" and st[VID_B]["status"] == "done"


def test_terminal_failure_is_recorded(roots, campaign):
    def crash(unit, attempt, force_offload):
        return 1                                           # child died without an error file
    rc = process(campaign / "worklist.json", 0, 1, campaign, campaign / "ckpt.pt", roots=roots,
                 clip_runner=crash)
    assert rc == 1
    err = json.loads((campaign / "errors" / f"{VID_A}.json").read_text())
    assert err["error_type"] == "SubprocessExit" and err["attempt"] == 1
    rows = {r["vid"]: r for r in ledger(load_worklist(campaign / "worklist.json"), campaign)}
    assert rows[VID_A]["status"] == "failed" and rows[VID_C]["status"] == "done"


def test_interrupted_clip_stays_pending_and_is_retried(roots, campaign):
    """A child killed by a signal (walltime SIGTERM, preemption) is an interruption, not a
    terminal failure: pending, retried on the next pass; errors/<vid>.json says why."""
    from vidstg_masks.worker import MAX_INTERRUPTIONS
    calls = []
    killed = {VID_A}

    def killed_once(unit, attempt, force_offload):
        calls.append((unit["vid"], attempt))
        if unit["vid"] in killed:
            killed.discard(unit["vid"])
            return -15                                     # SIGTERM, no error file written
        return fake_runner(roots, campaign, [])(unit, attempt, force_offload)
    rc = process(campaign / "worklist.json", 0, 1, campaign, campaign / "ckpt.pt", roots=roots,
                 clip_runner=killed_once)
    # the pass after the kill retries A inside the same call (still pending, unclaimed)
    assert calls[:1] == [(VID_A, 1)] and (VID_A, 2) in calls
    assert rc == 0
    rows = {r["vid"]: r for r in ledger(load_worklist(campaign / "worklist.json"), campaign)}
    assert rows[VID_A]["status"] == "done"
    # an interruption record on its own reads as pending, up to MAX_INTERRUPTIONS
    from vidstg_masks.records import write_json_atomic
    (campaign / "records" / f"{VID_A}.jsonl").unlink()
    err = dict(vid=VID_A, attempt=MAX_INTERRUPTIONS - 1, error_type="SubprocessExit",
               message="exit code -15 (killed by a signal)", cuda_oom=False, interrupted=True)
    write_json_atomic(campaign / "errors" / f"{VID_A}.json", err)
    unit = next(u for u in load_worklist(campaign / "worklist.json")["units"] if u["vid"] == VID_A)
    assert unit_status(unit, campaign)["status"] == "pending"
    write_json_atomic(campaign / "errors" / f"{VID_A}.json", dict(err, attempt=MAX_INTERRUPTIONS))
    assert unit_status(unit, campaign)["status"] == "failed"


def test_kill_after_oom_keeps_the_offload_flag(roots, campaign):
    from vidstg_masks.worker import EXIT_OOM
    from vidstg_masks.records import write_json_atomic
    seen = []

    def oom_then_killed_then_ok(unit, attempt, force_offload):
        seen.append((unit["vid"], attempt, force_offload))
        if unit["vid"] != VID_B:
            return fake_runner(roots, campaign, [])(unit, attempt, force_offload)
        if attempt == 1:
            write_json_atomic(campaign / "errors" / f"{VID_B}.json",
                              dict(vid=VID_B, attempt=1, error_type="OutOfMemoryError",
                                   message="CUDA out of memory", cuda_oom=True))
            return EXIT_OOM
        if attempt == 2:
            return -15                                     # killed during the offload retry
        return fake_runner(roots, campaign, [])(unit, attempt, force_offload)
    assert process(campaign / "worklist.json", 0, 1, campaign, campaign / "ckpt.pt", roots=roots,
                   clip_runner=oom_then_killed_then_ok) == 0
    b = [s for s in seen if s[0] == VID_B]
    assert b == [(VID_B, 1, False), (VID_B, 2, True), (VID_B, 3, True)]   # offload stays forced


def test_build_worklist_asserts_release_counts(roots, monkeypatch):
    from vidstg_masks import datasets
    from vidstg_masks.datasets import DataError
    build_worklist(roots, "val")                                        # fixture counts match
    monkeypatch.setitem(datasets.KNOWN_RECORDS_PER_SPLIT, "val", 999)
    with pytest.raises(DataError, match=r"VidSTG val: 2 records .* expected 999"):
        build_worklist(roots, "val")
    build_worklist(roots, "val", assert_counts=False)                   # escape hatch
    build_worklist(roots, "train")                                      # other splits unaffected
    monkeypatch.setitem(datasets.KNOWN_FACTS, "vidor_annotations", 1)
    with pytest.raises(DataError, match=r"VidOR annotations: 5 files"):
        build_worklist(roots, "train")


def test_roots_from_env_rebases_unit_paths(roots, data, tmp_path):
    """A worklist built under one set of roots runs under another (--roots-from-env)."""
    import shutil
    from vidstg_masks.datasets import Roots, rebase_unit_paths
    wl = build_worklist(roots, "all")
    moved = tmp_path / "moved"
    shutil.copytree(data["vidor_ann"], moved / "ann")
    shutil.copytree(data["videos"], moved / "videos")
    new = Roots(data["vidstg"], moved / "ann", moved / "videos", None)
    units = {u["vid"]: rebase_unit_paths(u, roots, new) for u in wl["units"]}
    assert units[VID_A]["vidor_ann"] == str(moved / "ann" / "training" / "0001" / f"{VID_A}.json")
    assert units[VID_A]["video_path"] == str(moved / "videos" / "0001" / f"{VID_A}.mp4")
    assert units[VID_D]["video_path"] is None                         # still missing
    # the old roots no longer exist: the worker must not touch them
    shutil.rmtree(data["vidor_ann"])
    shutil.rmtree(data["videos"])
    c = tmp_path / "camp"
    save_worklist(c / "worklist.json", wl)
    calls = []
    seen_paths = []

    def runner(unit, attempt, force_offload):
        seen_paths.append(unit["video_path"])
        return fake_runner(new, c, calls)(unit, attempt, force_offload)
    assert process(c / "worklist.json", 0, 1, c, c / "ckpt.pt", roots=new, clip_runner=runner,
                   roots_from_env=True) == 0
    assert sorted(v for v, _, _ in calls) == [VID_A, VID_B]
    assert all(p.startswith(str(moved)) for p in seen_paths)
    rows = {r["vid"]: r for r in ledger(load_worklist(c / "worklist.json"), c)}
    assert all(r["status"] == "done" for r in rows.values())


def test_campaign_status_counts(roots, campaign):
    from vidstg_masks.worker import campaign_status
    from vidstg_masks.records import write_json_atomic
    process(campaign / "worklist.json", 0, 1, campaign, campaign / "ckpt.pt", roots=roots,
            clip_runner=fake_runner(roots, campaign, []))
    (campaign / "records" / f"{VID_B}.jsonl").unlink()
    write_json_atomic(campaign / "errors" / f"{VID_B}.json",
                      dict(vid=VID_B, attempt=1, error_type="RuntimeError", message="x", cuda_oom=False))
    s = campaign_status(load_worklist(campaign / "worklist.json"), campaign)
    assert s["units"] == 4 and s["clips"] == {"done": 3, "failed": 1}
    assert s["masks"] > 0 and s["refusals"] > 0
    assert s["refusals_by_reason"]["too_many_objects"] == 17 * 12
    assert s["failed"] == [dict(vid=VID_B, attempts=1, error_type="RuntimeError")]
    assert s["gpu_wall_hours"] == round(1.0 / 3600, 3)


# ── gap rule, contained negatives, recorded settings ─────────────────────────

def test_clip_records_carry_contained_negatives_and_widen_anchor_fids(base_prov):
    from test_anchors import container_plan
    plan = container_plan()
    prompts = prompts_for_plan(plan)
    prov = add_contained_negatives(plan, prompts)
    m = np.ones((100, 100), bool)
    per_frame = {f: {t: (m, 1.0) for t in (0, 1)} for f in range(11)}
    recs, refusals = clip_records(plan, per_frame, base_prov, neg_prov=prov)
    assert refusals == {}
    by = {(r["tid"], r["fid"]): r for r in recs}
    pp0 = by[(0, 0)]["prompt_payload"]
    assert pp0["anchor_fids"] == [0, 2, 5, 8] and pp0["ref_anchor_fid"] == 0    # own anchors + co-prompts
    assert pp0["contained_negatives"] == prov[0]
    pp1 = by[(1, 2)]["prompt_payload"]
    assert pp1["anchor_fids"] == [2, 8] and pp1["contained_negatives"]["n_neg_clicks"] == 0
    for r in recs:
        validate_record(r)
    # without the option the payload is exactly what it always was
    plain, _ = clip_records(plan, per_frame, base_prov)
    assert plain[0]["prompt_payload"] == {"anchor_fids": [0, 5], "ref_anchor_fid": 0, "anchor_policy": "human"}


def test_subprocess_runner_forwards_the_policy_settings(roots, campaign, monkeypatch):
    import subprocess
    from vidstg_masks import sam_session, worker
    monkeypatch.setattr(sam_session, "assert_checkpoint", lambda p: "a" * 64)
    seen = []

    class Done:
        returncode = 0
    monkeypatch.setattr(subprocess, "run", lambda cmd, env: seen.append(cmd) or Done())
    unit = load_worklist(campaign / "worklist.json")["units"][0]
    run = worker.subprocess_runner(campaign / "worklist.json", campaign, campaign / "ckpt.pt",
                                   "human_gap", 8, hq_fallback="refuse", max_gap=30, gap_fill="any",
                                   keep_span_edges=True, contained_negatives=True)
    assert run(unit, 1, False) == 0
    cmd = seen[-1]
    assert cmd[1:4] == ["-m", "vidstg_masks.cli", "process-one"]
    for flag, value in [("--anchor-policy", "human_gap"), ("--max-anchors", "8"), ("--hq-fallback", "refuse"),
                        ("--max-gap", "30"), ("--gap-fill", "any"), ("--attempt", "1"),
                        ("--checkpoint-hash", "a" * 64)]:
        assert cmd[cmd.index(flag) + 1] == value, flag
    assert "--keep-span-edges" in cmd and "--contained-negatives" in cmd and "--force-offload" not in cmd
    assert cmd[cmd.index("--dispute-rule") + 1] == "refuse" and cmd[cmd.index("--dispute-winner") + 1] == "weak"
    assert "--dispute-score" not in cmd and cmd[cmd.index("--direction") + 1] == "forward"
    # defaults: span edges absent, negatives off sent explicitly (the child's own default is on),
    # the gap rule is the default one
    worker.subprocess_runner(campaign / "worklist.json", campaign, campaign / "ckpt.pt", "human", 16)(unit, 2, True)
    cmd = seen[-1]
    assert "--keep-span-edges" not in cmd and "--contained-negatives" not in cmd and "--force-offload" in cmd
    assert "--no-contained-negatives" in cmd
    assert cmd[cmd.index("--max-gap") + 1] == "60" and cmd[cmd.index("--gap-fill") + 1] == "human"
    # the passes and the dispute tie-break travel too
    worker.subprocess_runner(campaign / "worklist.json", campaign, campaign / "ckpt.pt", "human", 16,
                             direction="both", dispute_rule="higher_score", dispute_score=0.907,
                             dispute_winner="strong")(unit, 1, False)
    cmd = seen[-1]
    for flag, value in [("--direction", "both"), ("--dispute-rule", "higher_score"), ("--dispute-score", "0.907"),
                        ("--dispute-winner", "strong")]:
        assert cmd[cmd.index(flag) + 1] == value, flag


def test_run_clip_records_gap_fills_negatives_and_the_run_settings(roots, campaign, monkeypatch):
    """run_clip end to end on the CPU with the fake predictor standing in for SAM."""
    from conftest import FakePredictor
    from vidstg_masks import sam_session as ss
    from vidstg_masks.worker import campaign_status, run_clip
    monkeypatch.setattr(ss, "assert_checkpoint", lambda p: "b" * 64)
    monkeypatch.setattr(ss, "build_predictor", lambda checkpoint, n_obj: FakePredictor())
    monkeypatch.setattr(ss, "set_offload", lambda enabled: None)
    monkeypatch.setattr(ss, "cuda_reset_peak", lambda: None)
    monkeypatch.setattr(ss, "cuda_peak_gb", lambda: 0.0)
    monkeypatch.setattr(ss, "collect_sam2_scores", lambda predictor, sid: {})
    unit = next(u for u in load_worklist(campaign / "worklist.json")["units"] if u["vid"] == VID_B)
    run = run_clip(unit, roots, campaign, campaign / "ckpt.pt", "human_gap", 2, verbose=False,
                   max_gap=4, contained_negatives=True)
    assert run["status"] == "done" and run["objects"] == 2
    assert {k: run[k] for k in ("anchor_policy", "max_anchors", "hq_fallback", "max_gap", "gap_fill",
                                "keep_span_edges", "contained_negatives")} == dict(
        anchor_policy="human_gap", max_anchors=2, hq_fallback="least-flagged", max_gap=4,
        gap_fill="human", keep_span_edges=False, contained_negatives=True)
    recs = list(read_jsonl(campaign / "records" / f"{VID_B}.jsonl"))
    by = {(r["tid"], r["fid"]): r for r in recs}
    pp = by[(0, 0)]["prompt_payload"]
    assert by[(0, 0)]["prompt_mode"] == "pvs_box_multianchor"
    assert pp["anchor_policy"] == "human_gap" and pp["anchor_fids"] == [0, 3, 6, 9]
    assert pp["gap"]["n_gap_fills"] == 2 and pp["gap"]["baseline_fids"] == [0, 9]
    assert pp["contained_negatives"] == dict(neg_clicks={}, co_prompt_fids=[], co_prompt_tracker_fids=[],
                                             n_neg_clicks=0)          # nothing is contained in B
    for r in recs:
        validate_record(r)
    run_json = json.loads((campaign / "runs" / f"{VID_B}.json").read_text())
    assert run_json["contained_negatives"] is True and run_json["max_gap"] == 4
    # the same settings are recorded for a clip refused on the CPU, and status lists them once
    from vidstg_masks.records import base_provenance
    from vidstg_masks.worker import _refuse_on_cpu, run_settings
    settings = run_settings("human_gap", 2, max_gap=4, contained_negatives=True)
    unit_d = next(u for u in load_worklist(campaign / "worklist.json")["units"] if u["vid"] == VID_D)
    assert _refuse_on_cpu(unit_d, roots, campaign, "human_gap", 2, 2, base_provenance(code_commit="t"), settings)
    refused = json.loads((campaign / "runs" / f"{VID_D}.json").read_text())
    assert refused["reason"] == "video_missing" and refused["max_gap"] == 4 and refused["contained_negatives"] is True
    d_recs = list(read_jsonl(campaign / "records" / f"{VID_D}.jsonl"))
    assert all(r["prompt_payload"]["gap"] is None and r["prompt_payload"]["anchor_policy"] == "human_gap" for r in d_recs)
    for r in d_recs:
        validate_record(r)
    assert campaign_status(load_worklist(campaign / "worklist.json"), campaign)["settings"] == [settings]


class TwoPassFake:
    """FakePredictor whose propagation output depends on the requested direction."""

    def __init__(self, by_direction):
        from conftest import FakePredictor
        self._inner = FakePredictor()
        self.by_direction = by_direction

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def handle_stream_request(self, request):
        self._inner.requests.append(request)
        outs = self.by_direction[request["propagation_direction"]]
        for fid in sorted(outs):
            yield {"frame_index": fid, "outputs": outs[fid]}


def _outputs(masks: dict, probs=0.9):
    ids = sorted(masks)
    return {"out_obj_ids": ids, "out_binary_masks": [masks[i] for i in ids],
            "out_probs": [probs] * len(ids)}


def test_run_clip_both_directions_merges_and_records_the_rule(roots, campaign, monkeypatch):
    """--direction both: a forward and a backward session with the same prompts, merged frame
    by frame by the pixels rule; every record says what the merge did and why."""
    from vidstg_masks import sam_session as ss
    from vidstg_masks.records import rle_decode
    from vidstg_masks.worker import run_clip
    unit = next(u for u in load_worklist(campaign / "worklist.json")["units"] if u["vid"] == VID_B)
    plan, _ = plan_unit(unit, roots, 16, "human")
    tids = [t for t in plan["tids"] if plan["anchors"][t] is not None]
    assert len(tids) == 2
    t0, t1 = tids
    s0, s1 = plan["prop_span"]
    n = s1 - s0 + 1
    H, W = plan["H"], plan["W"]
    b = plan["boxes"][t1][s0 + 7]["bbox"]
    a = np.zeros((H, W), bool)                                   # t1: inside its own box
    a[int(b["ymin"]):int(b["ymax"]) + 1, int(b["xmin"]):int(b["xmax"]) + 1] = True
    body = np.zeros((H, W), bool)                                # t0: a block in a corner clear of t1
    for ys, xs in ((slice(0, 24), slice(0, 24)), (slice(H - 24, H), slice(W - 24, W)),
                   (slice(0, 24), slice(W - 24, W)), (slice(H - 24, H), slice(0, 24))):
        body[:] = False
        body[ys, xs] = True
        if not (body & a).any():
            break
    assert not (body & a).any()
    far = np.zeros((H, W), bool)                                 # a real mask somewhere else
    far[H // 2 - 5:H // 2 + 5, W // 2 - 5:W // 2 + 5] = True
    far &= ~(a | body)
    assert far.sum() >= 20
    speck = np.zeros((H, W), bool)
    speck[int(b["ymin"]):int(b["ymin"]) + 2, int(b["xmin"]):int(b["xmin"]) + 2] = True   # 4 px
    empty = np.zeros((H, W), bool)
    fwd = {s0 + i: _outputs({t0: body, t1: a}) for i in range(n - 1)}  # the forward pass stops short of the span end
    fwd[s0 + 5] = _outputs({t0: body, t1: empty})                # forward lost t1 at frame 5
    fwd[s0 + 7] = _outputs({t0: body, t1: far})                  # forward on something else at frame 7
    fwd[s0 + 8] = _outputs({t0: body | a, t1: speck})            # forward gave t1's pixels to t0 at frame 8
    fwd[s0 + 9] = _outputs({t0: body, t1: speck})                # forward kept a speck of t1 at frame 9
    bwd = {s0 + i: _outputs({t0: body, t1: a}, 0.8) for i in range(1, n)}  # the backward pass never predicts the span start
    bwd[s0 + 8] = _outputs({t0: body | a, t1: a}, 0.8)
    monkeypatch.setattr(ss, "assert_checkpoint", lambda p: "c" * 64)
    monkeypatch.setattr(ss, "build_predictor", lambda checkpoint, n_obj: TwoPassFake(dict(forward=fwd, backward=bwd)))
    monkeypatch.setattr(ss, "set_offload", lambda enabled: None)
    monkeypatch.setattr(ss, "cuda_reset_peak", lambda: None)
    monkeypatch.setattr(ss, "cuda_peak_gb", lambda: 0.0)
    calls = []

    def scores(predictor, sid):                                  # forward session 0.9, backward 0.8
        calls.append(1)
        return {(f, t): (0.9 if len(calls) % 2 == 1 else 0.8) for f in range(s0, s1 + 1) for t in (t0, t1)}
    monkeypatch.setattr(ss, "collect_sam2_scores", scores)
    run = run_clip(unit, roots, campaign, campaign / "ckpt.pt", "human", 16, verbose=False,
                   contained_negatives=False, direction="both")
    assert run["status"] == "done" and run["direction"] == "both" and run["frames"] == 2 * n - 2
    assert (run["agree_iou"], run["speck_floor"], run["speck_ratio"], run["dispute_score"]) == (0.3, 20, 0.1, None)
    assert (run["dispute_rule"], run["dispute_winner"]) == ("refuse", "weak")
    recs = list(read_jsonl(campaign / "records" / f"{VID_B}.jsonl"))
    by = {(r["tid"], r["fid"]): r for r in recs}
    mg = {k: r["prompt_payload"]["merge"] for k, r in by.items() if r["prompt_payload"].get("merge")}
    assert by[(t0, s0 + 3)]["prompt_payload"]["direction"] == "bidirectional"
    assert mg[(t0, s0)]["rule"] == "forward_only" and mg[(t0, s1)]["rule"] == "backward_only"
    assert mg[(t1, s0 + 5)]["rule"] == "backward_only" and by[(t1, s0 + 5)]["rle"] is not None
    assert by[(t1, s0 + 5)]["mask_confidence"] == 0.8           # the backward pass's own confidence
    assert mg[(t0, s0 + 3)]["rule"] == "agree" and by[(t0, s0 + 3)]["mask_confidence"] == 0.9   # forward kept
    # frame 7: two real masks that do not overlap -> refused, no box consulted
    assert mg[(t1, s0 + 7)]["rule"] == "dispute" and mg[(t1, s0 + 7)]["decision"] == "refused"
    assert by[(t1, s0 + 7)]["reason_code"] == "disputed_mask" and by[(t1, s0 + 7)]["rle"] is None
    # frame 8: t0 agrees with itself but holds t1's pixels; t1's only candidate is backward -> handover
    assert mg[(t0, s0 + 8)]["rule"] == "agree" and mg[(t0, s0 + 8)]["decision"] == "forward"
    assert mg[(t0, s0 + 8)]["handover"]["removed_px"] == int(a.sum()) and mg[(t0, s0 + 8)]["handover"]["to"] == [t1]
    assert rle_decode(by[(t0, s0 + 8)]["rle"]).sum() == int(body.sum())
    assert mg[(t1, s0 + 8)]["rule"] == "forward_speck" and by[(t1, s0 + 8)]["mask_confidence"] == 0.8
    assert not (rle_decode(by[(t0, s0 + 8)]["rle"]) & rle_decode(by[(t1, s0 + 8)]["rle"])).any()
    # frame 9: a 4 px forward speck against a real backward mask -> backward
    assert mg[(t1, s0 + 9)]["rule"] == "forward_speck" and mg[(t1, s0 + 9)]["source"] == "backward"
    assert run["merge"]["dispute"] == 1 and run["merge"]["disputed_mask"] == 1 and run["merge"]["forward_speck"] == 2
    assert run["merge"]["decision:refused"] == 1 and run["refusals"]["disputed_mask"] == 1
    assert run["merge"]["forward_only"] == 2 and run["merge"]["backward_only"] == 3
    for r in recs:
        validate_record(r)


def test_run_clip_backward_alone_records_the_direction(roots, campaign, monkeypatch):
    from conftest import FakePredictor
    from vidstg_masks import sam_session as ss
    from vidstg_masks.worker import run_clip
    unit = next(u for u in load_worklist(campaign / "worklist.json")["units"] if u["vid"] == VID_B)
    pred = FakePredictor()
    monkeypatch.setattr(ss, "assert_checkpoint", lambda p: "c" * 64)
    monkeypatch.setattr(ss, "build_predictor", lambda checkpoint, n_obj: pred)
    monkeypatch.setattr(ss, "set_offload", lambda enabled: None)
    monkeypatch.setattr(ss, "cuda_reset_peak", lambda: None)
    monkeypatch.setattr(ss, "cuda_peak_gb", lambda: 0.0)
    monkeypatch.setattr(ss, "collect_sam2_scores", lambda predictor, sid: {})
    run = run_clip(unit, roots, campaign, campaign / "ckpt.pt", "human", 16, verbose=False,
                   contained_negatives=False, direction="backward")
    prop = [r for r in pred.requests if r["type"] == "propagate_in_video"]
    assert len(prop) == 1 and prop[0]["propagation_direction"] == "backward"
    assert prop[0]["start_frame_index"] == run["prop_span"][1]
    recs = list(read_jsonl(campaign / "records" / f"{VID_B}.jsonl"))
    assert recs and all(r["prompt_payload"]["direction"] == "backward" for r in recs)
    assert all(r["reason_code"] == "empty_mask" for r in recs if r["prompt_payload"]["anchor_fids"])  # the fake predicted nothing
    with pytest.raises(ValueError):
        run_clip(unit, roots, campaign, campaign / "ckpt.pt", "human", 16, verbose=False, direction="sideways")
