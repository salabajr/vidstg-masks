#!/usr/bin/env bash
# One machine, one GPU: doctor -> build-worklist -> process (shard 0/1) -> export.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
: "${VIDSTG_ROOT:?set VIDSTG_ROOT}" "${VIDOR_ANN_ROOT:?set VIDOR_ANN_ROOT}" "${VIDOR_VIDEO_ROOT:?set VIDOR_VIDEO_ROOT}"
SPLIT="${SPLIT:-val}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${REPO_ROOT}/outputs/${SPLIT}}"
SAM31_CHECKPOINT="${SAM31_CHECKPOINT:-${REPO_ROOT}/checkpoints/sam3.1/sam3.1_multiplex.pt}"
ANCHOR_POLICY="${ANCHOR_POLICY:-human_gap}"
MAX_ANCHORS="${MAX_ANCHORS:-16}"
# Policy settings (docs/PIPELINE.md, "Anchor policies"); the defaults are the package's own.
MAX_GAP="${MAX_GAP:-60}"
GAP_FILL="${GAP_FILL:-human}"
CONTAINED_NEGATIVES="${CONTAINED_NEGATIVES:-1}"
# Passes and merge (README, Settings that change the masks); the defaults are the measured tie-break
# (higher_score / 0.907 / strong); DIRECTION=forward DISPUTE_RULE=refuse is the forward pass alone.
DIRECTION="${DIRECTION:-both}"
AGREE_IOU="${AGREE_IOU:-0.3}"
SPECK_FLOOR="${SPECK_FLOOR:-20}"
SPECK_RATIO="${SPECK_RATIO:-0.1}"
DISPUTE_RULE="${DISPUTE_RULE:-higher_score}"
DISPUTE_SCORE="${DISPUTE_SCORE:-}"
DISPUTE_WINNER="${DISPUTE_WINNER:-strong}"
policy_flags=()
[[ "${KEEP_SPAN_EDGES:-0}" == "1" ]] && policy_flags+=(--keep-span-edges)
# passed either way, so the variable decides: 1 -> on, anything else -> off
if [[ "${CONTAINED_NEGATIVES}" == "1" ]]; then policy_flags+=(--contained-negatives); else policy_flags+=(--no-contained-negatives); fi
policy_flags+=(--direction "${DIRECTION}" --agree-iou "${AGREE_IOU}" --speck-floor "${SPECK_FLOOR}" --speck-ratio "${SPECK_RATIO}"
               --dispute-rule "${DISPUTE_RULE}" --dispute-winner "${DISPUTE_WINNER}")
[[ -n "${DISPUTE_SCORE}" ]] && policy_flags+=(--dispute-score "${DISPUTE_SCORE}")
export VIDSTG_ROOT VIDOR_ANN_ROOT VIDOR_VIDEO_ROOT VIDOR_TRANSCODED_ROOT="${VIDOR_TRANSCODED_ROOT:-}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${CAMPAIGN_ROOT}"

"${PYTHON_BIN}" -m vidstg_masks.cli doctor --checkpoint "${SAM31_CHECKPOINT}" ${DOCTOR_FLAGS:-}

extra=()
[[ -n "${VIDS:-}" ]] && extra+=(--vids ${VIDS})
[[ -n "${VIDS_FILE:-}" ]] && extra+=(--vids-file "${VIDS_FILE}")
[[ -n "${LIMIT:-}" ]] && extra+=(--limit "${LIMIT}")
[[ "${ASSERT_COUNTS:-1}" == "0" ]] && extra+=(--no-assert-counts)
"${PYTHON_BIN}" -m vidstg_masks.cli build-worklist --split "${SPLIT}" --campaign-root "${CAMPAIGN_ROOT}" \
  --if-missing ${extra[@]+"${extra[@]}"}

"${PYTHON_BIN}" -m vidstg_masks.cli process \
  --worklist "${CAMPAIGN_ROOT}/worklist.json" --campaign-root "${CAMPAIGN_ROOT}" \
  --checkpoint "${SAM31_CHECKPOINT}" --shard-index 0 --shard-count 1 \
  --anchor-policy "${ANCHOR_POLICY}" --max-anchors "${MAX_ANCHORS}" --hq-fallback "${HQ_FALLBACK:-least-flagged}" \
  --max-gap "${MAX_GAP}" --gap-fill "${GAP_FILL}" ${policy_flags[@]+"${policy_flags[@]}"}

"${PYTHON_BIN}" -m vidstg_masks.cli export \
  --worklist "${CAMPAIGN_ROOT}/worklist.json" --campaign-root "${CAMPAIGN_ROOT}"

# ConCor Video tables (export/concor/); CAPTIONS=captions.jsonl adds the full records
concor=()
[[ -n "${CAPTIONS:-}" ]] && concor+=(--captions "${CAPTIONS}")
"${PYTHON_BIN}" -m vidstg_masks.cli export-concor \
  --worklist "${CAMPAIGN_ROOT}/worklist.json" --campaign-root "${CAMPAIGN_ROOT}" ${concor[@]+"${concor[@]}"}
