import json
from pathlib import Path
import shutil

import pytest

from conftest import H, N_FRAMES, VID_A, VID_B, W
from test_worker import fake_runner
from vidstg_masks import cli
from vidstg_masks.worker import build_worklist, process, save_worklist


def run(argv):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    return e.value.code


def test_build_worklist_and_plan_cli(data, tmp_path, capsys):
    c = tmp_path / "camp"
    assert run(["build-worklist", "--split", "all", "--campaign-root", str(c), "--vids", VID_A, VID_B]) == 0
    wl = json.loads((c / "worklist.json").read_text())
    assert [u["vid"] for u in wl["units"]] == [VID_A, VID_B]
    assert run(["plan", "--worklist", str(c / "worklist.json"), "--vids", VID_A]) == 0
    out = capsys.readouterr().out
    assert "ref fid 6" in out and "REFUSED (no_human_keyframe)" in out


def test_doctor_reports_synthetic_counts_as_mismatch(data, capsys):
    rc = run(["doctor", "--skip-hash", "--skip-gpu-libs"])
    out = capsys.readouterr().out
    assert "known fact vidstg_records: 20 (expected 44,808)" in out
    assert "[OK  ] vidstg_root" in out
    assert rc == 1                                        # counts do not match the real corpus


def test_doctor_missing_roots(monkeypatch, capsys):
    for k in ("VIDSTG_ROOT", "VIDOR_ANN_ROOT", "VIDOR_VIDEO_ROOT"):
        monkeypatch.delenv(k, raising=False)
    assert run(["doctor", "--skip-hash", "--skip-facts", "--skip-gpu-libs"]) == 1


def test_process_shard_bounds(data, tmp_path):
    c = tmp_path / "camp"
    save_worklist(c / "worklist.json", build_worklist(__import__("vidstg_masks.datasets").datasets.Roots.from_env(), "val"))
    with pytest.raises(SystemExit):
        cli.main(["process", "--worklist", str(c / "worklist.json"), "--campaign-root", str(c),
                  "--checkpoint", str(c / "x.pt"), "--shard-index", "2", "--shard-count", "2"])


def test_export_cli_exit_code_follows_integrity(roots, tmp_path):
    c = tmp_path / "camp"
    save_worklist(c / "worklist.json", build_worklist(roots, "train"))
    process(c / "worklist.json", 0, 1, c, c / "x.pt", roots=roots, clip_runner=fake_runner(roots, c, []))
    assert run(["export", "--worklist", str(c / "worklist.json"), "--campaign-root", str(c)]) == 0
    (c / "records" / f"{VID_A}.jsonl").write_text("")
    assert run(["export", "--worklist", str(c / "worklist.json"), "--campaign-root", str(c)]) == 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_render_cli_writes_h264(roots, tmp_path):
    c = tmp_path / "camp"
    save_worklist(c / "worklist.json", build_worklist(roots, "train"))
    process(c / "worklist.json", 0, 1, c, c / "x.pt", roots=roots, clip_runner=fake_runner(roots, c, []))
    out = tmp_path / "ov.mp4"
    assert run(["render", "--vid", VID_A, "--campaign-root", str(c), "--out", str(out)]) == 0
    assert out.stat().st_size > 0
    from vidstg_masks.video import decode_check, probe_size
    assert decode_check(out, 12)["frame_count_ok"]
    w, h = probe_size(out)
    assert w == W and h > H                                 # the banner strip sits under the frame


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_render_side_by_side_doubles_the_width(roots, tmp_path):
    c = tmp_path / "camp"
    save_worklist(c / "worklist.json", build_worklist(roots, "train"))
    process(c / "worklist.json", 0, 1, c, c / "x.pt", roots=roots, clip_runner=fake_runner(roots, c, []))
    out = tmp_path / "sbs.mp4"
    assert run(["render", "--vid", VID_A, "--campaign-root", str(c), "--out", str(out),
                "--side-by-side", "--crf", "28", "--no-labels"]) == 0
    from vidstg_masks.video import decode_check, probe_size
    assert decode_check(out, N_FRAMES)["frame_count_ok"]
    w, h = probe_size(out)
    assert w == 2 * W and h > H


def test_status_cli(roots, tmp_path, capsys):
    c = tmp_path / "camp"
    save_worklist(c / "worklist.json", build_worklist(roots, "all"))
    process(c / "worklist.json", 0, 1, c, c / "x.pt", roots=roots, clip_runner=fake_runner(roots, c, []))
    assert run(["status", "--worklist", str(c / "worklist.json"), "--campaign-root", str(c)]) == 0
    out = capsys.readouterr().out
    assert "done 4 · failed 0 · pending 0" in out and "too_many_objects 204" in out
    assert run(["status", "--worklist", str(c / "worklist.json"), "--campaign-root", str(c), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["clips"] == {"done": 4}


def test_build_worklist_cli_count_assertion(data, tmp_path, monkeypatch, capsys):
    from vidstg_masks import datasets
    c = tmp_path / "camp"
    monkeypatch.setitem(datasets.KNOWN_RECORDS_PER_SPLIT, "val", 999)
    with pytest.raises(datasets.DataError):
        cli.main(["build-worklist", "--split", "val", "--campaign-root", str(c)])
    assert not (c / "worklist.json").exists()
    assert run(["build-worklist", "--split", "val", "--campaign-root", str(c), "--no-assert-counts"]) == 0
    assert (c / "worklist.json").exists()


def test_build_worklist_if_missing_keeps_the_existing_file(data, tmp_path, capsys):
    c = tmp_path / "camp"
    assert run(["build-worklist", "--split", "val", "--campaign-root", str(c)]) == 0
    before = (c / "worklist.json").read_bytes()
    assert run(["build-worklist", "--split", "train", "--campaign-root", str(c), "--if-missing"]) == 0
    assert "kept" in capsys.readouterr().out
    assert (c / "worklist.json").read_bytes() == before


def test_policy_flags_reach_the_worker(data, tmp_path, monkeypatch):
    from vidstg_masks import worker
    c = tmp_path / "camp"
    assert run(["build-worklist", "--split", "val", "--campaign-root", str(c)]) == 0
    seen = {}

    def stub(*args, **kw):
        seen.clear()
        seen.update(kw, args=args)
        return 0
    monkeypatch.setattr(worker, "install_signal_handlers", lambda: None)
    monkeypatch.setattr(worker, "process", stub)
    monkeypatch.setattr(worker, "process_one_main", stub)
    common = ["--worklist", str(c / "worklist.json"), "--campaign-root", str(c), "--checkpoint", str(c / "x.pt")]
    assert run(["process", *common, "--anchor-policy", "human_gap", "--max-gap", "0", "--gap-fill", "any",
                "--keep-span-edges", "--contained-negatives"]) == 0
    assert seen["args"][5:7] == ("human_gap", 16)
    assert {k: seen[k] for k in ("max_gap", "gap_fill", "keep_span_edges", "contained_negatives")} == dict(
        max_gap=0, gap_fill="any", keep_span_edges=True, contained_negatives=True)
    assert run(["process", *common]) == 0              # defaults: human_gap, negatives on, gap rule 60 / human
    assert seen["args"][5:7] == ("human_gap", 16)
    assert {k: seen[k] for k in ("max_gap", "gap_fill", "keep_span_edges", "contained_negatives")} == dict(
        max_gap=60, gap_fill="human", keep_span_edges=False, contained_negatives=True)
    assert run(["process", *common, "--no-contained-negatives"]) == 0      # the opt-out reaches the worker
    assert seen["args"][5:7] == ("human_gap", 16) and seen["contained_negatives"] is False
    assert run(["process", *common, "--anchor-policy", "human"]) == 0      # the earlier default, by name
    assert seen["args"][5:7] == ("human", 16) and seen["contained_negatives"] is True
    assert run(["process-one", *common, "--vid", VID_B, "--anchor-policy", "hq", "--max-gap", "30",
                "--keep-span-edges"]) == 0
    assert seen["args"][4:6] == ("hq", 16) and seen["max_gap"] == 30 and seen["keep_span_edges"] is True
    assert seen["gap_fill"] == "human" and seen["contained_negatives"] is True
    assert run(["process-one", *common, "--vid", VID_B, "--no-contained-negatives"]) == 0
    assert seen["args"][4:6] == ("human_gap", 16) and seen["contained_negatives"] is False
    with pytest.raises(SystemExit):                                        # unknown policy rejected
        cli.main(["process", *common, "--anchor-policy", "any"])


def test_plan_cli_describes_gap_fills_and_contained_negatives(data, tmp_path, capsys):
    c = tmp_path / "camp"
    assert run(["build-worklist", "--split", "val", "--campaign-root", str(c)]) == 0
    assert run(["plan", "--worklist", str(c / "worklist.json"), "--vids", VID_B, "--anchor-policy", "human_gap",
                "--max-anchors", "2", "--max-gap", "4", "--contained-negatives"]) == 0
    out = capsys.readouterr().out
    assert "policy human_gap" in out
    assert "gap rule (max gap 4, fill human): +2 fills (3, 6) over baseline [0, 9]" in out
    assert "contained negatives: 0 negative clicks, 0 co-prompts (0 at tracker boxes)" in out
    # hq with the coverage-rule flags still prints the gate summary
    assert run(["plan", "--worklist", str(c / "worklist.json"), "--vids", VID_B, "--anchor-policy", "hq",
                "--max-gap", "0", "--keep-span-edges"]) == 0
    assert "hq tid 0:" in capsys.readouterr().out


def test_direction_flags_reach_the_worker(data, tmp_path, monkeypatch):
    from vidstg_masks import worker
    c = tmp_path / "camp"
    assert run(["build-worklist", "--split", "val", "--campaign-root", str(c)]) == 0
    seen = {}

    def stub(*args, **kw):
        seen.clear()
        seen.update(kw, args=args)
        return 0
    monkeypatch.setattr(worker, "install_signal_handlers", lambda: None)
    monkeypatch.setattr(worker, "process", stub)
    monkeypatch.setattr(worker, "process_one_main", stub)
    common = ["--worklist", str(c / "worklist.json"), "--campaign-root", str(c), "--checkpoint", str(c / "x.pt")]
    assert run(["process", *common]) == 0                                  # defaults: both passes, the measured tie-break
    assert (seen["direction"], seen["agree_iou"], seen["speck_floor"], seen["speck_ratio"], seen["dispute_score"]) == ("both", 0.3, 20, 0.1, 0.907)
    assert (seen["dispute_rule"], seen["dispute_winner"]) == ("higher_score", "strong")
    assert run(["process", *common, "--direction", "forward", "--dispute-rule", "refuse"]) == 0   # the forward pass alone
    assert (seen["direction"], seen["dispute_rule"], seen["dispute_score"], seen["dispute_winner"]) == ("forward", "refuse", None, "strong")
    assert run(["process", *common, "--dispute-rule", "forward_score"]) == 0                     # a score rule takes the default threshold
    assert (seen["dispute_rule"], seen["dispute_score"]) == ("forward_score", 0.907)
    assert run(["process", *common, "--direction", "forward", "--agree-iou", "0.8", "--speck-floor", "30",
                "--speck-ratio", "0.2", "--dispute-rule", "higher_score", "--dispute-score", "0.907",
                "--dispute-winner", "strong"]) == 0
    assert (seen["direction"], seen["agree_iou"], seen["speck_floor"], seen["speck_ratio"], seen["dispute_score"]) == ("forward", 0.8, 30, 0.2, 0.907)
    assert (seen["dispute_rule"], seen["dispute_winner"]) == ("higher_score", "strong")
    with pytest.raises(SystemExit):                                              # a threshold without a rule
        cli.main(["process", *common, "--dispute-score", "0.9"])
    with pytest.raises(SystemExit):                                              # a rule without a threshold
        cli.main(["process-one", *common, "--vid", VID_B, "--dispute-rule", "forward_score"])
    assert run(["process-one", *common, "--vid", VID_B, "--direction", "backward"]) == 0
    assert seen["direction"] == "backward" and seen["speck_floor"] == 20
    with pytest.raises(SystemExit):
        cli.main(["process", *common, "--direction", "sideways"])


def test_transcode_requeues_the_clips_it_repairs(data, tmp_path, monkeypatch):
    """A clip refused as decode_black (or frame_size_mismatch) has records, so the worker counts it as
    done; `transcode --worklist --campaign-root` moves those files to superseded/ once the re-encode
    matches the annotation, and the clip is pending again. A unit refused for another reason is left alone."""
    from vidstg_masks import video
    from vidstg_masks.worker import load_worklist, unit_status
    c = tmp_path / "camp"
    assert run(["build-worklist", "--split", "val", "--campaign-root", str(c)]) == 0
    wl = load_worklist(c / "worklist.json")
    vids = [u["vid"] for u in wl["units"]]
    (c / "runs").mkdir(); (c / "records").mkdir(); (c / "errors").mkdir()
    (c / "runs" / f"{vids[0]}.json").write_text(json.dumps({"vid": vids[0], "status": "refused", "reason": "decode_black"}))
    (c / "records" / f"{vids[0]}.jsonl").write_text("{}\n")
    (c / "errors" / f"{vids[0]}.json").write_text("{}")
    (c / "runs" / f"{vids[1]}.json").write_text(json.dumps({"vid": vids[1], "status": "refused", "reason": "too_many_objects"}))
    (c / "records" / f"{vids[1]}.jsonl").write_text("{}\n")
    assert unit_status(wl["units"][0], c)["status"] == "done"
    calls = []
    monkeypatch.setattr(video, "transcode_h264", lambda src, dst: calls.append((src, dst)) or shutil.copy(src, dst.parent.mkdir(parents=True, exist_ok=True) or dst))
    monkeypatch.setattr(video, "probe_frame_count", lambda p: N_FRAMES)
    monkeypatch.setattr(video, "probe_size", lambda p: (W, H))
    monkeypatch.setattr(video, "is_black", lambda p: False)
    out = tmp_path / "transcoded"
    assert run(["transcode", "--worklist", str(c / "worklist.json"), "--campaign-root", str(c),
                "--vidor-transcoded-root", str(out)]) == 0
    assert [v for s_, d_ in calls for v in [s_.stem]] == [vids[0]]          # only the decode_black unit
    assert not (c / "runs" / f"{vids[0]}.json").exists() and not (c / "records" / f"{vids[0]}.jsonl").exists()
    assert (c / "superseded" / "runs" / f"{vids[0]}.json").is_file() and (c / "superseded" / "records" / f"{vids[0]}.jsonl").is_file()
    assert (c / "superseded" / "errors" / f"{vids[0]}.json").is_file()
    assert unit_status(wl["units"][0], c)["status"] == "pending"
    assert (c / "runs" / f"{vids[1]}.json").is_file() and unit_status(wl["units"][1], c)["status"] == "done"
    # the worklist now names the repaired file: the worker reads the unit's frozen video_path, not the roots
    wl2 = load_worklist(c / "worklist.json")
    u0 = next(u for u in wl2["units"] if u["vid"] == vids[0])
    assert Path(u0["video_path"]).resolve() == (out / Path(wl["units"][0]["video_path"]).relative_to(data["videos"])).resolve()
    assert next(u for u in wl2["units"] if u["vid"] == vids[1])["video_path"] == wl["units"][1]["video_path"]
    # a re-encode that does not match the annotation leaves the refusal in place and fails the command
    monkeypatch.setattr(video, "probe_size", lambda p: (W + 1, H))
    (c / "runs" / f"{vids[0]}.json").write_text(json.dumps({"vid": vids[0], "status": "refused", "reason": "frame_size_mismatch"}))
    (c / "records" / f"{vids[0]}.jsonl").write_text("{}\n")
    assert run(["transcode", "--worklist", str(c / "worklist.json"), "--campaign-root", str(c),
                "--vidor-transcoded-root", str(out)]) == 1
    assert (c / "runs" / f"{vids[0]}.json").is_file()
