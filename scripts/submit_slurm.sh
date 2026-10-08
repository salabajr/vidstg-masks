#!/usr/bin/env bash
# Check the dataset (doctor) and build the worklist on the login node, then submit an N-task GPU
# array plus a dependent CPU export job (export, then export-concor; CAPTIONS=captions.jsonl adds the full ConCor records). Re-running with the same CAMPAIGN_ROOT resumes: finished
# clips are skipped, interrupted ones retried. Every job id is appended to CAMPAIGN_ROOT/submissions.txt.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
: "${VIDSTG_ROOT:?set VIDSTG_ROOT}" "${VIDOR_ANN_ROOT:?set VIDOR_ANN_ROOT}" "${VIDOR_VIDEO_ROOT:?set VIDOR_VIDEO_ROOT}"
SLURM_PARTITION="${SLURM_PARTITION:?set SLURM_PARTITION}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"                 # passed as --account only when set
GPU_GRES="${GPU_GRES:-gpu:1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SPLIT="${SPLIT:-val}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${REPO_ROOT}/outputs/${SPLIT}}"
SAM31_REPO_ROOT="${SAM31_REPO_ROOT:-${REPO_ROOT}/external/sam3}"
SAM31_CHECKPOINT="${SAM31_CHECKPOINT:-${REPO_ROOT}/checkpoints/sam3.1/sam3.1_multiplex.pt}"
ANCHOR_POLICY="${ANCHOR_POLICY:-human_gap}"
MAX_ANCHORS="${MAX_ANCHORS:-16}"
MAX_GAP="${MAX_GAP:-60}"
GAP_FILL="${GAP_FILL:-human}"
KEEP_SPAN_EDGES="${KEEP_SPAN_EDGES:-0}"
CONTAINED_NEGATIVES="${CONTAINED_NEGATIVES:-1}"   # 1 -> --contained-negatives, anything else -> --no-contained-negatives
# Passes and merge (README, Settings that change the masks): DIRECTION both (the default) runs the backward pass
# too and merges; the defaults are the measured tie-break (higher_score / 0.907 / strong);
# DIRECTION=forward DISPUTE_RULE=refuse is the forward pass alone.
DIRECTION="${DIRECTION:-both}"
AGREE_IOU="${AGREE_IOU:-0.3}"
SPECK_FLOOR="${SPECK_FLOOR:-20}"
SPECK_RATIO="${SPECK_RATIO:-0.1}"
DISPUTE_RULE="${DISPUTE_RULE:-higher_score}"
DISPUTE_SCORE="${DISPUTE_SCORE:-}"
DISPUTE_WINNER="${DISPUTE_WINNER:-strong}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
SIGNAL_LEAD_SECONDS="${SIGNAL_LEAD_SECONDS:-1800}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-48G}"
VIDSTG_MASKS_CACHE_ROOT="${VIDSTG_MASKS_CACHE_ROOT:-${REPO_ROOT}/.cache}"
STAGE_CHECKPOINT="${STAGE_CHECKPOINT:-1}"
ROOTS_FROM_ENV="${ROOTS_FROM_ENV:-0}"
(( NUM_WORKERS >= 1 )) || { echo "NUM_WORKERS must be at least 1" >&2; exit 2; }

export VIDSTG_ROOT VIDOR_ANN_ROOT VIDOR_VIDEO_ROOT VIDOR_TRANSCODED_ROOT="${VIDOR_TRANSCODED_ROOT:-}"
export VIDSTG_MASKS_CACHE_ROOT STAGE_CHECKPOINT ROOTS_FROM_ENV
export HF_HUB_CACHE="${HF_HUB_CACHE:-${VIDSTG_MASKS_CACHE_ROOT}/huggingface/hub}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${CAMPAIGN_ROOT}/slurm_logs"

# Roots, ffmpeg, python modules and the dataset counts, before anything is submitted. The
# checkpoint hash and the GPU libraries are checked by the array tasks themselves.
"${PYTHON_BIN}" -m vidstg_masks.cli doctor --skip-hash --skip-gpu-libs --checkpoint "${SAM31_CHECKPOINT}" \
  --sam3-repo "${SAM31_REPO_ROOT}" ${DOCTOR_FLAGS:-}

extra=()
[[ -n "${VIDS:-}" ]] && extra+=(--vids ${VIDS})
[[ -n "${VIDS_FILE:-}" ]] && extra+=(--vids-file "${VIDS_FILE}")
[[ -n "${LIMIT:-}" ]] && extra+=(--limit "${LIMIT}")
[[ "${ASSERT_COUNTS:-1}" == "0" ]] && extra+=(--no-assert-counts)
"${PYTHON_BIN}" -m vidstg_masks.cli build-worklist --split "${SPLIT}" --campaign-root "${CAMPAIGN_ROOT}" \
  --if-missing ${extra[@]+"${extra[@]}"}

account_flag=()
[[ -n "${SLURM_ACCOUNT}" ]] && account_flag=(--account="${SLURM_ACCOUNT}")
array_end=$((NUM_WORKERS - 1))
worker_job="$(sbatch --parsable \
  --partition="${SLURM_PARTITION}" ${account_flag[@]+"${account_flag[@]}"} --gres="${GPU_GRES}" \
  --time="${TIME_LIMIT}" --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY}" \
  --signal="B:USR1@${SIGNAL_LEAD_SECONDS}" \
  --array="0-${array_end}" \
  --output="${CAMPAIGN_ROOT}/slurm_logs/worker-%A_%a.out" \
  --error="${CAMPAIGN_ROOT}/slurm_logs/worker-%A_%a.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",PYTHON_BIN="${PYTHON_BIN}",CAMPAIGN_ROOT="${CAMPAIGN_ROOT}",SAM31_REPO_ROOT="${SAM31_REPO_ROOT}",SAM31_CHECKPOINT="${SAM31_CHECKPOINT}",SHARD_COUNT="${NUM_WORKERS}",ANCHOR_POLICY="${ANCHOR_POLICY}",MAX_ANCHORS="${MAX_ANCHORS}",HQ_FALLBACK="${HQ_FALLBACK:-least-flagged}",MAX_GAP="${MAX_GAP}",GAP_FILL="${GAP_FILL}",KEEP_SPAN_EDGES="${KEEP_SPAN_EDGES}",CONTAINED_NEGATIVES="${CONTAINED_NEGATIVES}",DIRECTION="${DIRECTION}",AGREE_IOU="${AGREE_IOU}",SPECK_FLOOR="${SPECK_FLOOR}",SPECK_RATIO="${SPECK_RATIO}",DISPUTE_RULE="${DISPUTE_RULE}",DISPUTE_SCORE="${DISPUTE_SCORE}",DISPUTE_WINNER="${DISPUTE_WINNER}" \
  "${REPO_ROOT}/slurm/process_array.slurm")"

export_job="$(sbatch --parsable \
  --partition="${EXPORT_PARTITION:-${SLURM_PARTITION}}" ${account_flag[@]+"${account_flag[@]}"} \
  --time="${EXPORT_TIME_LIMIT:-02:00:00}" --cpus-per-task=4 --mem="${EXPORT_MEMORY:-32G}" \
  --dependency="afterany:${worker_job}" \
  --output="${CAMPAIGN_ROOT}/slurm_logs/export-%j.out" \
  --error="${CAMPAIGN_ROOT}/slurm_logs/export-%j.err" \
  --wrap="export PYTHONPATH='${PYTHONPATH}'; '${PYTHON_BIN}' -m vidstg_masks.cli export --worklist '${CAMPAIGN_ROOT}/worklist.json' --campaign-root '${CAMPAIGN_ROOT}' && '${PYTHON_BIN}' -m vidstg_masks.cli export-concor --worklist '${CAMPAIGN_ROOT}/worklist.json' --campaign-root '${CAMPAIGN_ROOT}'${CAPTIONS:+ --captions '${CAPTIONS}'}")"

printf '%s worker_array=%s export=%s split=%s workers=%s direction=%s dispute=%s/%s/%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "${worker_job}" "${export_job}" "${SPLIT}" "${NUM_WORKERS}" "${DIRECTION}" "${DISPUTE_RULE}" "${DISPUTE_SCORE:-default}" "${DISPUTE_WINNER}" >> "${CAMPAIGN_ROOT}/submissions.txt"
printf 'worker_array=%s\nexport=%s\ncampaign=%s\nprogress: %s -m vidstg_masks.cli status --worklist %s --campaign-root %s\n' \
  "${worker_job}" "${export_job}" "${CAMPAIGN_ROOT}" "${PYTHON_BIN}" "${CAMPAIGN_ROOT}/worklist.json" "${CAMPAIGN_ROOT}"
