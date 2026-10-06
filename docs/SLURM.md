# Slurm

## Full-corpus command

Every variable named. The same set is in `examples/all.env`
(`set -a; source examples/all.env; set +a; bash scripts/submit_slurm.sh`).

```bash
VIDSTG_ROOT=/data/vidstg \
VIDOR_ANN_ROOT=/data/vidor/annotation \
VIDOR_VIDEO_ROOT=/data/vidor/video \
VIDOR_TRANSCODED_ROOT=/data/vidor/video_h264 \
SAM31_REPO_ROOT=$PWD/external/sam3 \
SAM31_CHECKPOINT=$PWD/checkpoints/sam3.1/sam3.1_multiplex.pt \
PYTHON_BIN=$PWD/.venv/bin/python \
VIDSTG_MASKS_CACHE_ROOT=/scratch/$USER/vidstg-masks-cache \
SLURM_PARTITION=gpu SLURM_ACCOUNT=your-account GPU_GRES=gpu:a40:1 \
NUM_WORKERS=16 TIME_LIMIT=12:00:00 CPUS_PER_TASK=8 MEMORY=48G \
SIGNAL_LEAD_SECONDS=1800 STAGE_CHECKPOINT=1 \
SPLIT=all CAMPAIGN_ROOT=/outputs/vidstg-masks-all \
bash scripts/submit_slurm.sh
```

Required: `SLURM_PARTITION`, `VIDSTG_ROOT`, `VIDOR_ANN_ROOT`, `VIDOR_VIDEO_ROOT`;
`SLURM_ACCOUNT` where the site needs one (passed as `--account` only when set). Defaults: `GPU_GRES` `gpu:1` (use the site's form, e.g. `gpu:a40:1`),
`NUM_WORKERS` 8, `TIME_LIMIT` 12:00:00, `CPUS_PER_TASK` 8, `MEMORY` 48G,
`SIGNAL_LEAD_SECONDS` 1800, `STAGE_CHECKPOINT` 1, `VIDSTG_MASKS_CACHE_ROOT` `<repo>/.cache`,
`SAM31_REPO_ROOT` `<repo>/external/sam3`, `SAM31_CHECKPOINT`
`<repo>/checkpoints/sam3.1/sam3.1_multiplex.pt`, `PYTHON_BIN` `<repo>/.venv/bin/python`,
`SPLIT` val, `CAMPAIGN_ROOT` `<repo>/outputs/<SPLIT>`, `VIDOR_TRANSCODED_ROOT` unset.
Optional: `ANCHOR_POLICY` (human_gap), `MAX_ANCHORS` (16), `HQ_FALLBACK` (least-flagged),
`MAX_GAP` (60), `GAP_FILL` (human), `KEEP_SPAN_EDGES` (1 adds `--keep-span-edges`),
`CONTAINED_NEGATIVES` (1, the default, passes `--contained-negatives`; any other value passes
`--no-contained-negatives`), `DIRECTION` (forward; `both` adds the backward pass and the merge),
`AGREE_IOU` (0.3), `SPECK_FLOOR` (20), `SPECK_RATIO` (0.1), `DISPUTE_RULE` (refuse),
`DISPUTE_SCORE` (unset; the threshold of `higher_score` / `forward_score`), `DISPUTE_WINNER`
(weak) — the measured tie-break is `DIRECTION=both DISPUTE_RULE=higher_score DISPUTE_SCORE=0.907
DISPUTE_WINNER=strong` (README, Backward pass and the merge), `VIDS`, `VIDS_FILE`, `LIMIT`,
`WORKLIST_PATH` (point the workers at a subset worklist inside the same campaign),
`ROOTS_FROM_ENV` (1 adds `--roots-from-env` to `process`, see Resuming), `ASSERT_COUNTS`
(0 adds `--no-assert-counts` to `build-worklist`), `DOCTOR_FLAGS`, `EXPORT_PARTITION`
(= `SLURM_PARTITION`), `EXPORT_TIME_LIMIT` (02:00:00), `EXPORT_MEMORY` (32G).

## Job DAG

```
login node (CPU)                    GPU array: one task = one GPU = one shard              CPU, afterany
doctor --skip-hash --skip-gpu-libs
  roots, ffmpeg, dataset counts
        |
build-worklist --split all  ---->  process_array.slurm [0 .. NUM_WORKERS-1]  ---------->  export
  worklist.json (counts asserted)    stage checkpoint, sha256 once per task                 masks.parquet
  submissions.txt                    claim clip -> process-one -> records/<vid>.jsonl       refusals.parquet
                                     USR1 at T - SIGNAL_LEAD_SECONDS: drain, exit 99,      ledger.csv
                                     scontrol requeue -> the same task id resumes           manifest.json
```

A failing `doctor` (a missing root, a count that does not match) stops the script before any
`sbatch`. The worklist is built only if `$CAMPAIGN_ROOT/worklist.json` does not exist. The
export job depends on the array with `afterany`: it runs once the array ends, over whatever
is done by then, exits 1 if the integrity check finds a problem, and can be re-run by hand
at any time (`vidstg-masks export --worklist ... --campaign-root ...`).

## At the walltime

`submit_slurm.sh` passes `--signal=B:USR1@${SIGNAL_LEAD_SECONDS}` (default 1800 s). On `USR1`
the shard stops claiming clips, finishes the active clip if it can, exits 99, and
`process_array.slurm` runs `scontrol requeue` on itself; the requeued task appends to the
same log file and continues from the campaign state on disk. A clip still running when Slurm
kills the task at the limit, or under preemption or `scancel`, is recorded in
`errors/<vid>.json` with `"interrupted": true`, stays `pending`, and is retried on the next
pass, 5 attempts in total. It is not counted as failed. The lead should exceed the longest clip:
from the Cost table (README), a clip takes about 1.5 min plus 0.4 s per frame per object, so
a lead of L seconds covers a clip up to about (L - 90) / 0.4 frame-objects (4,275 at the
default); raise `SIGNAL_LEAD_SECONDS` when a unit's `n_frames x n_tids` in `worklist.json`
is larger than that.

## Watching progress

```bash
vidstg-masks status --worklist $CAMPAIGN_ROOT/worklist.json --campaign-root $CAMPAIGN_ROOT
```

prints clips done / failed / pending (with the number of interrupted clips awaiting retry),
masks and refusals so far, refusals by reason code, GPU wall-hours so far, and the failed
vids with their `error_type`; `--json` gives the same as one JSON object. It writes nothing
and runs on the login node; `submit_slurm.sh` prints the exact command when it exits.
`$CAMPAIGN_ROOT/slurm_logs/worker-<array>_<task>.out` is one file per task across requeues
(`--open-mode=append`); its lines start `[clip]`, `[committed]`, `[interrupted]`, `[refused]`,
`[oom]`, `[failed]`, `[drain]`. `$CAMPAIGN_ROOT/submissions.txt` gets one line per
`submit_slurm.sh` run, `<utc time> worker_array=<id> export=<id> split=<split> workers=<n>`,
for `squeue -j` and `sacct -j`.

## Resuming

Run `submit_slurm.sh` again with the same `CAMPAIGN_ROOT`. The existing worklist is reused,
done clips (`records/<vid>.jsonl`) are skipped, pending clips are claimed and run, and a new
export job is queued behind the new array. A task that finds nothing pending exits 0. To
retry a failed clip, delete its `errors/<vid>.json` and resume; to rerun a done clip (for
example after `transcode`), delete its `records/<vid>.jsonl` and `runs/<vid>.json` first.

`worklist.json` carries the dataset roots and the absolute annotation and video path of every
unit as they were on the node that built it. If the compute nodes mount the data elsewhere,
or the data moved after the build, submit with `ROOTS_FROM_ENV=1`: `process_array.slurm` then
passes `--roots-from-env` to `process`, which takes the roots from the job environment
(`submit_slurm.sh` exports them) and rebases every unit's paths onto them. Without it every
clip fails with `FileNotFoundError`.

## Staging and caches

Per task, `process_array.slurm`:

- puts the Hugging Face download cache (`HF_HUB_CACHE`) under `VIDSTG_MASKS_CACHE_ROOT`
  (default `<repo>/.cache`), the same place `setup.sh` and `download_model.sh` put
  `PIP_CACHE_DIR` and `HF_HUB_CACHE` (the worker downloads nothing; this keeps the
  variable set identical everywhere);
- points the mutable caches (`XDG_CACHE_HOME`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`,
  `CUDA_CACHE_PATH`, `PYTHONPYCACHEPREFIX`) at a per-job scratch directory
  `${SLURM_TMPDIR:-/tmp}/vidstg-masks-<job id>` (`TMPDIR` when set and `SLURM_TMPDIR` is
  not) that is removed when the task exits;
- with `STAGE_CHECKPOINT=1` (default) copies the 3.5 GB checkpoint once to
  `${SLURM_TMPDIR:-/tmp}/vidstg-masks/` under an `flock`, so several tasks on one node copy
  it once (a task waits up to 1800 s for the lock), when at least 5 GiB is free there;
  otherwise, or if the copy fails, `SAM31_CHECKPOINT` is read in place. The sha256 is
  computed once per task, not once per clip, and passed to every per-clip subprocess
  (`process-one --checkpoint-hash`). `STAGE_CHECKPOINT=0` always reads the shared path.

None of these caches lands in `$HOME`. On shared clusters `$HOME` is often quota-limited, and
a full `$HOME` is the usual first failure on a new machine; put `VIDSTG_MASKS_CACHE_ROOT` on
scratch or project storage. The worker also sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless already set.

## Sizing

Measured on one RTX A5000 24 GB (README, Cost): about 1.5 min of model load per clip plus
about 0.4 s per frame per object; 451 val clips about 92 GPU-h; all 602 val videos about
120 GPU-h; all 6,770 videos about 1,250-3,100 GPU-h depending on object counts; peak VRAM
6-19 GB per clip. Host memory per task stayed under 20 GB in our runs; the video is decoded
to CPU memory (`offload_video_to_cpu`), so long 1080p clips need more.

`counts.prop_frame_objects` in `worklist.json` is the campaign's total frame-object count:
multiply by 0.4 s and add 1.5 min per unit for the GPU-hour budget. One array pass of
`NUM_WORKERS=16` tasks at `TIME_LIMIT=12:00:00` is 192 GPU-h, so the val split fits in one
pass and the full corpus needs 7 to 17 passes (1,250 / 192 to 3,100 / 192), each requeue
adding one model-free restart per task. More tasks shorten the calendar time in proportion
as long as the site runs them concurrently.

[OPEN] Times on other GPUs: Roy measures them from `status` (GPU wall-hours) after the first
pass on his cards; the A5000 numbers are the only ones measured. [OPEN] Whether the export
job's defaults (2 h, 32 GB) hold for the full corpus: decided by the first full export; the
export is re-runnable by hand.

## Portability

No site paths are hard-coded. The worker script needs the exported variables above plus
`PYTHON_BIN`. Change `GPU_GRES` to the site's form (`gpu:a100:1`, `gpu:h100:1`, ...) without
code changes; one task needs one GPU with 24 GB or more.
