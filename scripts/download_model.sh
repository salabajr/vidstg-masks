#!/usr/bin/env bash
# Download the gated SAM 3.1 Object Multiplex checkpoint and verify its sha256.
# Run `hf auth login` first with an account that has accepted the facebook/sam3.1 gate.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/sam3.1}"
VIDSTG_MASKS_CACHE_ROOT="${VIDSTG_MASKS_CACHE_ROOT:-${REPO_ROOT}/.cache}"
# The download cache goes off $HOME; the token written by `hf auth login` stays where the CLI
# keeps it ($HF_HOME/token, default ~/.cache/huggingface), or comes from HF_TOKEN.
export HF_HUB_CACHE="${HF_HUB_CACHE:-${VIDSTG_MASKS_CACHE_ROOT}/huggingface/hub}"
mkdir -p "${HF_HUB_CACHE}"
EXPECTED_SHA256="0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"

mkdir -p "${CHECKPOINT_DIR}"
"${VENV_PATH}/bin/hf" download facebook/sam3.1 sam3.1_multiplex.pt --local-dir "${CHECKPOINT_DIR}"

checkpoint="${CHECKPOINT_DIR}/sam3.1_multiplex.pt"
actual="$(sha256sum "${checkpoint}" | awk '{print $1}')"
if [[ "${actual}" != "${EXPECTED_SHA256}" ]]; then
  echo "checkpoint sha256 mismatch: got ${actual}, expected ${EXPECTED_SHA256}" >&2
  echo "refusing to keep an unverified checkpoint; removing ${checkpoint}" >&2
  rm -f "${checkpoint}"
  exit 1
fi
printf 'checkpoint=%s\nsha256=%s\n' "${checkpoint}" "${actual}"
