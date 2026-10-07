"""slurm/process_array.slurm around a stub worker: the drain signal sent to the batch shell
reaches the worker, the task requeues itself on the worker's exit 99, and the worker's exit code
is the task's otherwise. No Slurm, no GPU: `scontrol` is a stub on the PATH."""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "slurm" / "process_array.slurm"

STUB_WORKER = """#!/bin/bash
# stands in for `python -m vidstg_masks.cli process`: on USR1 it exits 99 like the shard does
trap 'exit 99' USR1
echo "$$" > "$STUB_UP"
for i in $(seq 1 "${STUB_TICKS:-100}"); do sleep 0.1; done
exit "${STUB_RC:-0}"
"""


def _setup(tmp_path, **stub_env):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "scontrol").write_text(f'#!/bin/bash\necho "$@" >> "{tmp_path / "scontrol.log"}"\n')
    (bin_ / "nvidia-smi").write_text("#!/bin/bash\nexit 1\n")
    worker = tmp_path / "worker.sh"
    worker.write_text(STUB_WORKER)
    for f in (bin_ / "scontrol", bin_ / "nvidia-smi", worker):
        f.chmod(0o755)
    (tmp_path / "camp").mkdir()
    env = dict(os.environ)
    for k in ("SLURM_TMPDIR", "SLURM_ARRAY_TASK_ID"):
        env.pop(k, None)
    env.update(PATH=f"{bin_}:{env['PATH']}", REPO_ROOT=str(tmp_path), PYTHON_BIN=str(worker),
               CAMPAIGN_ROOT=str(tmp_path / "camp"), SAM31_REPO_ROOT=str(tmp_path),
               SAM31_CHECKPOINT=str(tmp_path / "x.pt"), SHARD_COUNT="1", VIDSTG_ROOT=str(tmp_path),
               VIDOR_ANN_ROOT=str(tmp_path), VIDOR_VIDEO_ROOT=str(tmp_path), STAGE_CHECKPOINT="0",
               SLURM_JOB_ID="4242", TMPDIR=str(tmp_path), VIDSTG_MASKS_CACHE_ROOT=str(tmp_path / "cache"),
               STUB_UP=str(tmp_path / "up"), **stub_env)
    return env


def _run(tmp_path, env, send=None):
    proc = subprocess.Popen(["bash", str(SCRIPT)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if send is not None:
        deadline = time.time() + 10
        while not (tmp_path / "up").exists() and time.time() < deadline:
            time.sleep(0.05)
        assert (tmp_path / "up").exists(), "the worker never started"
        proc.send_signal(send)
    out, _ = proc.communicate(timeout=30)
    return proc.returncode, out


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_drain_signal_reaches_the_worker_and_the_task_requeues(tmp_path):
    rc, out = _run(tmp_path, _setup(tmp_path), send=signal.SIGUSR1)
    assert rc == 0, out
    assert "signal USR1: forwarding to the worker" in out and "drained on signal; requeueing" in out
    assert (tmp_path / "scontrol.log").read_text().strip() == "requeue 4242"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_worker_exit_code_is_the_task_exit_code(tmp_path):
    rc, out = _run(tmp_path, _setup(tmp_path, STUB_TICKS="2"))
    assert rc == 0, out
    assert not (tmp_path / "scontrol.log").exists()
    failing = tmp_path / "b"
    failing.mkdir()
    rc, out = _run(failing, _setup(failing, STUB_TICKS="2", STUB_RC="1"))
    assert rc == 1, out
    assert not (failing / "scontrol.log").exists()
