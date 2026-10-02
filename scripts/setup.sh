#!/usr/bin/env bash
# Create the virtual environment, install the package, torch, and the pinned sam3 checkout.
# INSTALL_GPU=0 installs the package and the test extras only (login node, CPU-only checkout).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
INSTALL_GPU="${INSTALL_GPU:-1}"
# Caches off $HOME (quota-limited on most shared clusters); override VIDSTG_MASKS_CACHE_ROOT.
VIDSTG_MASKS_CACHE_ROOT="${VIDSTG_MASKS_CACHE_ROOT:-${REPO_ROOT}/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${VIDSTG_MASKS_CACHE_ROOT}/pip}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${VIDSTG_MASKS_CACHE_ROOT}/huggingface/hub}"   # downloads; the token stays under HF_HOME
mkdir -p "${PIP_CACHE_DIR}" "${HF_HUB_CACHE}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
SAM31_REPO_ROOT="${SAM31_REPO_ROOT:-${REPO_ROOT}/external/sam3}"
SAM3_COMMIT="96914d2425f90a64f45ca977c2b5165418099543"

"${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else "Python 3.11+ is required")'
"${PYTHON_BIN}" -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_PATH}/bin/python" -m pip install -e "${REPO_ROOT}[test]"

if [[ "${INSTALL_GPU}" != "1" ]]; then
  printf 'environment=%s\ninstall_gpu=0 (no torch, no sam3; tests, doctor --skip-gpu-libs, export and status work)\n' "${VENV_PATH}"
  exit 0
fi

if ! "${VENV_PATH}/bin/python" -c 'import torch' >/dev/null 2>&1; then
  "${VENV_PATH}/bin/python" -m pip install "torch==${TORCH_VERSION}" torchvision \
    --index-url "${TORCH_INDEX_URL}"
fi

if [[ ! -f "${SAM31_REPO_ROOT}/pyproject.toml" ]]; then
  mkdir -p "$(dirname "${SAM31_REPO_ROOT}")"
  git clone https://github.com/facebookresearch/sam3.git "${SAM31_REPO_ROOT}"
fi
git -C "${SAM31_REPO_ROOT}" fetch --quiet origin "${SAM3_COMMIT}" || true
git -C "${SAM31_REPO_ROOT}" checkout --quiet "${SAM3_COMMIT}"
"${VENV_PATH}/bin/python" -m pip install -e "${SAM31_REPO_ROOT}"

printf 'environment=%s\nsam3_source=%s\nsam3_commit=%s\n' \
  "${VENV_PATH}" "${SAM31_REPO_ROOT}" "$(git -C "${SAM31_REPO_ROOT}" rev-parse HEAD)"
