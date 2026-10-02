"""torch and sam3 must never be imported by the package itself (only inside GPU functions),
so every CPU command works on a login node. Checked in a fresh interpreter so the order of
the other tests cannot mask a stray import."""

import subprocess
import sys
from pathlib import Path

MODULES = ["vidstg_masks", "vidstg_masks.anchors", "vidstg_masks.cli", "vidstg_masks.datasets",
           "vidstg_masks.export", "vidstg_masks.records", "vidstg_masks.render",
           "vidstg_masks.sam3_compat", "vidstg_masks.sam_session", "vidstg_masks.video",
           "vidstg_masks.worker"]


def test_package_imports_without_torch_or_sam3():
    code = ("import sys, importlib\n"
            + "".join(f"importlib.import_module({m!r})\n" for m in MODULES)
            + "assert 'torch' not in sys.modules, 'torch imported'\n"
            "assert 'sam3' not in sys.modules, 'sam3 imported'\n"
            "print('clean')\n")
    src = Path(__file__).resolve().parents[1] / "src"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "clean"


def test_tests_themselves_did_not_import_torch():
    # the fixtures and CPU code paths exercised by this suite must not need torch either
    assert "sam3" not in sys.modules
