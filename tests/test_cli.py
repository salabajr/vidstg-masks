import json
import shutil

import pytest

from conftest import VID_A, VID_B
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
    from vidstg_masks.video import decode_check
    assert decode_check(out, 12)["frame_count_ok"]


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
    assert run(["process", *common]) == 0                                  # default: one forward pass
    assert (seen["direction"], seen["agree_iou"], seen["dispute_iou"], seen["refuse_disputed"]) == ("forward", 0.7, 0.3, False)
    assert run(["process", *common, "--direction", "both", "--agree-iou", "0.8", "--dispute-iou", "0.2",
                "--refuse-disputed"]) == 0
    assert (seen["direction"], seen["agree_iou"], seen["dispute_iou"], seen["refuse_disputed"]) == ("both", 0.8, 0.2, True)
    assert run(["process-one", *common, "--vid", VID_B, "--direction", "backward"]) == 0
    assert seen["direction"] == "backward" and seen["refuse_disputed"] is False
    with pytest.raises(SystemExit):
        cli.main(["process", *common, "--direction", "sideways"])
