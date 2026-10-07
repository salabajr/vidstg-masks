#!/usr/bin/env bash
#SBATCH --job-name=vidstg-masks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@1800
#SBATCH --open-mode=append
# One array task = one GPU = one `vidstg-masks process` shard. Every task walks the whole
# worklist from its own offset and claims videos with exclusive lease files, so any task count
# balances itself. USR1 (SIGNAL_LEAD_SECONDS before the time limit; submit_slurm.sh sets it,
# 30 min here as the fallback) is forwarded to the shard, which stops claiming clips, finishes
# the active one if it can and exits 99; this script then requeues the task. A clip still
# running when Slurm kills the task at the limit is recorded as interrupted and retried after
# the requeue.
set -euo pipefail
: "${REPO_ROOT:?}" "${PYTHON_BIN:?}" "${CAMPAIGN_ROOT:?}" "${SAM31_REPO_ROOT:?}" "${SAM31_CHECKPOINT:?}" "${SHARD_COUNT:?}"
: "${VIDSTG_ROOT:?}" "${VIDOR_ANN_ROOT:?}" "${VIDOR_VIDEO_ROOT:?}"
task_id="${SLURM_ARRAY_TASK_ID:-0}"
export PYTHONPATH="${REPO_ROOT}/src:${SAM31_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

# Caches: nothing under $HOME (quota-limited on most shared clusters). Persistent ones under the
# cache root; mutable per-job ones (torch/triton/CUDA compile caches, .pyc) on node-local scratch,
# removed when the task exits.
VIDSTG_MASKS_CACHE_ROOT="${VIDSTG_MASKS_CACHE_ROOT:-${REPO_ROOT}/.cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${VIDSTG_MASKS_CACHE_ROOT}/huggingface/hub}"
node_scratch="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
job_scratch="${node_scratch}/vidstg-masks-${SLURM_JOB_ID:-$$}"
mkdir -p "${job_scratch}" "${HF_HUB_CACHE}"
export XDG_CACHE_HOME="${job_scratch}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${job_scratch}/inductor"
export TRITON_CACHE_DIR="${job_scratch}/triton"
export CUDA_CACHE_PATH="${job_scratch}/cuda"
export PYTHONPYCACHEPREFIX="${job_scratch}/pycache"
trap 'rm -rf "${job_scratch}"' EXIT

# Checkpoint staging: copy the 3.5 GB file to node-local storage once per node (flock: several
# tasks on one node copy it once), when at least 5 GiB is free there; otherwise read the shared
# path. The worker hashes whichever file it is given once, before its first clip.
checkpoint="${SAM31_CHECKPOINT}"
if [[ "${STAGE_CHECKPOINT:-1}" == "1" ]]; then
  stage_dir="${node_scratch}/vidstg-masks"
  staged="${stage_dir}/$(basename "${SAM31_CHECKPOINT}")"
  mkdir -p "${stage_dir}"
  free_kib="$(df -Pk "${stage_dir}" | awk 'NR==2{print $4}')"
  if (( free_kib >= 5 * 1024 * 1024 )); then
    (
      flock -w 1800 9 || exit 1
      if [[ ! -s "${staged}" ]] || (( $(stat -c %s "${staged}") != $(stat -c %s "${SAM31_CHECKPOINT}") )); then
        rm -f "${staged}.part"
        cp "${SAM31_CHECKPOINT}" "${staged}.part" && mv "${staged}.part" "${staged}"
      fi
    ) 9>"${stage_dir}/.lock" && checkpoint="${staged}" \
      || echo "[task ${task_id}] staging failed; reading the checkpoint from ${SAM31_CHECKPOINT}"
  else
    echo "[task ${task_id}] under 5 GiB free in ${stage_dir}; reading the checkpoint from ${SAM31_CHECKPOINT}"
  fi
fi

echo "[task ${task_id}/${SHARD_COUNT}] host=$(hostname) job=${SLURM_JOB_ID} restart=${SLURM_RESTART_COUNT:-0} checkpoint=${checkpoint}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

roots_flag=()
[[ "${ROOTS_FROM_ENV:-0}" == "1" ]] && roots_flag=(--roots-from-env)
policy_flags=()
[[ "${KEEP_SPAN_EDGES:-0}" == "1" ]] && policy_flags+=(--keep-span-edges)
# passed either way, so the variable decides: 1 (the default) -> on, anything else -> off
if [[ "${CONTAINED_NEGATIVES:-1}" == "1" ]]; then policy_flags+=(--contained-negatives); else policy_flags+=(--no-contained-negatives); fi
# passes and merge; DISPUTE_SCORE is the threshold of a score rule (unset under refuse)
policy_flags+=(--direction "${DIRECTION:-forward}" --agree-iou "${AGREE_IOU:-0.3}" --speck-floor "${SPECK_FLOOR:-20}"
               --speck-ratio "${SPECK_RATIO:-0.1}" --dispute-rule "${DISPUTE_RULE:-refuse}" --dispute-winner "${DISPUTE_WINNER:-weak}")
[[ -n "${DISPUTE_SCORE:-}" ]] && policy_flags+=(--dispute-score "${DISPUTE_SCORE}")

# The drain signal (`--signal=B:USR1@...`: USR1 to this batch shell, SIGNAL_LEAD_SECONDS before
# the limit) and a TERM reach this shell, not the worker. Forward them, and keep waiting for the
# worker afterwards: bash's `wait` returns as soon as a trap has run. Without the trap the shell
# dies on USR1 (exit code 10), the worker is killed with the job step, and nothing is requeued.
worker_pid=""
forward() {
  echo "[task ${task_id}] signal $1: forwarding to the worker"
  [[ -n "${worker_pid}" ]] && kill -s "$1" "${worker_pid}" 2>/dev/null || true
}
trap 'forward USR1' USR1
trap 'forward TERM' TERM
set +e
"${PYTHON_BIN}" -m vidstg_masks.cli process \
  --worklist "${WORKLIST_PATH:-${CAMPAIGN_ROOT}/worklist.json}" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --checkpoint "${checkpoint}" \
  --shard-index "${task_id}" --shard-count "${SHARD_COUNT}" \
  --anchor-policy "${ANCHOR_POLICY:-human_gap}" --max-anchors "${MAX_ANCHORS:-16}" \
  --hq-fallback "${HQ_FALLBACK:-least-flagged}" --max-gap "${MAX_GAP:-60}" --gap-fill "${GAP_FILL:-human}" \
  ${policy_flags[@]+"${policy_flags[@]}"} ${roots_flag[@]+"${roots_flag[@]}"} &
worker_pid=$!
wait "${worker_pid}"
worker_rc=$?
while kill -0 "${worker_pid}" 2>/dev/null; do
  wait "${worker_pid}"
  worker_rc=$?
done
set -e
if [[ "${worker_rc}" -eq 99 ]]; then
  echo "[task ${task_id}] drained on signal; requeueing"
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
exit "${worker_rc}"
