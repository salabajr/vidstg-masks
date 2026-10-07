# Code structure

What each module owns, how one clip travels through them, where a change belongs, and what
the tests cover. The package is `src/vidstg_masks/`, about 4,200 lines; `torch` and `sam3` are
imported inside functions of `sam_session.py` and `sam3_compat.py` (and, guarded, in `doctor`'s
GPU-library check), so every other module
runs on a CPU-only machine (`tests/test_no_torch.py` checks that).

## The modules

| module | lines | owns | key functions |
|---|---|---|---|
| `cli.py` | 576 | the `vidstg-masks` command: the parser, one `cmd_*` function per subcommand, the dispute-settings check, the transcode re-queue | `main`, `cmd_doctor`, `cmd_build_worklist`, `cmd_process`, `cmd_process_one`, `cmd_status`, `cmd_export`, `cmd_export_concor`, `cmd_render`, `cmd_transcode`, `requeue_refused_unit` |
| `datasets.py` | 251 | reading VidSTG and VidOR, the dataset roots (`Roots`), the known counts, finding a clip's video | `load_vidstg`, `build_vidor_index`, `load_vidor`, `boxes_for_tid`, `resolve_video`, `rebase_unit_paths`, `assert_known_counts`, `known_facts_check` |
| `anchors.py` | 295 | the plan of a clip from the annotations alone: object set, spans, anchors, prompts as relative boxes, contained negatives | `plan_clip`, `apply_anchor_policy`, `prompts_for_plan`, `add_contained_negatives`, `in_span_fids`, `describe_plan` |
| `anchor_quality.py` | 456 | the gap rule (`_fill_gaps`) shared by `human_gap` and `hq`, and the `hq` quality gate over keyframes (edge, overlap, motion, size, blur) | `human_gap_plan`, `apply_hq_policy`, `hq_plan_for`, `keyframe_metrics`, `Thresholds` |
| `video.py` | 113 | probing and decoding checks, the H.264 transcode | `decode_check`, `is_black`, `probe_frame_count`, `probe_size`, `transcode_h264` |
| `worker.py` | 810 | the worklist, the claims, the shard loop, the per-clip subprocess, pre-checks, the records of a clip, the run summary, status and ledger | `build_worklist`, `claim`, `process`, `subprocess_runner`, `process_one_main`, `run_clip`, `precheck_clip`, `plan_unit`, `clip_records`, `unit_status`, `campaign_status`, `ledger` |
| `sam_session.py` | 179 | the SAM 3.1 session: build the predictor from the checkpoint, prompt every object, propagate, collect scores; the state-offload decision | `build_predictor`, `run_session`, `needs_state_offload`, `checkpoint_sha256`, `assert_checkpoint` |
| `sam3_compat.py` | 202 | the patches the vendored sam3 commit needs for box-prompted video sessions (kwarg filtering, refined masks without detector cache, state offload, buffer purging, a device fix) | `apply_pvs_patches`, `collect_sam2_scores`, `purge_prompt_buffers`, `set_state_offload` |
| `merge.py` | 320 | the merge of the forward and the backward pass: readings, the pixels rule, the dispute tie-break, the handover | `merge_passes`, `merge_frame`, `read_candidates`, `decide_pixels`, `tiebreak_pick`, `check_dispute_settings` |
| `records.py` | 218 | the record contract: provenance fields, reason codes, COCO RLE, atomic JSONL and JSON, validation, the code commit | `make_record`, `base_provenance`, `rle_encode`, `rle_decode`, `write_jsonl_atomic`, `write_json_atomic`, `validate_record`, `git_commit` |
| `export.py` | 200 | the Parquet tables, the ledger, the manifest, the integrity check | `export_campaign`, `check_clip` |
| `concor.py` | 457 | the ConCor Video export: records, tables, their validator's rules | `export_concor`, `relation_record`, `validate_record`, `read_captions` |
| `render.py` | 152 | the QA video | `render_overlay` |
| `__init__.py` | 14 | the pins: model name, SAM version, sam3 commit, checkpoint sha256 | `MODEL_NAME`, `SAM_VERSION`, `SAM3_COMMIT`, `CHECKPOINT_SHA256` |

Around the package:

```
schema/mask_record.schema.json   the record contract as JSON Schema (export validates against it)
scripts/setup.sh                 virtual environment, the package, torch, the pinned sam3 checkout; caches off $HOME
scripts/download_model.sh        the gated checkpoint, sha256 verified
scripts/run_local.sh             doctor -> build-worklist -> process (one shard) -> export -> export-concor
scripts/submit_slurm.sh          doctor -> build-worklist -> sbatch array (slurm/process_array.slurm) -> dependent export job
slurm/process_array.slurm        one array task: caches on node scratch, checkpoint staged once per node, process, requeue on drain
examples/*.env                   complete variable sets for a smoke test, the val split, the whole corpus
tests/                           pytest, CPU only
docs/                            this documentation
```

## The path of one clip

```
cli.cmd_process
└─ worker.process                      the shard loop: shard_order, unit_status, claim
   ├─ worker._refuse_on_cpu            too_many_objects / video_missing: refusal records, no GPU
   └─ worker._run_unit                 attempts, OOM retry with offload, interruption bookkeeping
      └─ worker.subprocess_runner      `vidstg-masks process-one` in a fresh process
         └─ cli.cmd_process_one -> worker.process_one_main -> worker.run_clip
            ├─ worker.plan_unit        anchors.plan_clip: the human plan
            ├─ worker.precheck_clip    video.decode_check: frame count, size, black
            ├─ anchors.apply_anchor_policy   anchor_quality.human_gap_plan / hq_plan_for, after the pre-checks
            ├─ anchors.prompts_for_plan + anchors.add_contained_negatives
            ├─ sam_session.run_session(direction="forward")      masks + scores per frame and object
            ├─ sam_session.run_session(direction="backward")     --direction both or backward only
            ├─ merge.merge_passes                                 both only: one mask per object and frame, or a refusal
            ├─ worker.clip_records     records.make_record for every (tid, fid) with a box in the span
            └─ records.write_jsonl_atomic -> records/<vid>.jsonl; runs/<vid>.json
```

Later, on a CPU: `export.export_campaign` reads `records/*.jsonl` and the VidOR annotations and
writes `export/`; `concor.export_concor` writes `export/concor/`; `render.render_overlay`
decodes the video again and paints the records.

### What each step guarantees

- `build_worklist` writes nothing unless the annotation set carries the release counts
  (`datasets.assert_known_counts`), so a wrong dataset is caught before any GPU work.
- `claim` creates the lease with `O_EXCL`; two shards never run the same clip. A lease older
  than 30 minutes (no heartbeat) is taken over.
- `precheck_clip` compares the decoded frame count and size with the annotation; a mismatch
  refuses the clip. Nothing rescales an index.
- `run_session` sends the box corners of every anchor and the negative clicks in the same
  `add_prompt`, purges the detector buffers between scattered prompt frames, propagates once,
  and always closes the session (`finally`).
- `merge_passes` never reads a box and never leaves two written masks sharing a pixel.
- `clip_records` writes exactly one record per boxed (tid, fid) in the span, mask or refusal,
  each with the full provenance set; `export` re-derives that set from the annotation and
  checks it.
- `write_jsonl_atomic` writes to `.part`, fsyncs and renames, so a half-written clip is never
  mistaken for a finished one.

## Where to change what

| you want to | change | and |
|---|---|---|
| add a command-line flag that changes the masks | `cli._add_policy` or `_add_runner`; `worker.run_settings` (so it lands in `runs/<vid>.json`); the `process` / `process-one` plumbing in `worker.run_clip`, `process_one_main`, `subprocess_runner` | the three launchers (an environment variable each), `README.md` Settings, `docs/CLI.md`, `docs/SLURM.md`; a `tests/test_worker.py` case that the subprocess command carries it |
| change which keyframes become anchors | `anchors.plan_clip` (the human plan), `anchor_quality._fill_gaps` (the gap rule), `anchor_quality.apply_hq_policy` (the gate) | `prompt_payload` fields in `records.py` / the schema if new provenance is recorded; `tests/test_anchors.py`, `tests/test_anchor_quality.py` |
| change the prompts sent to SAM | `anchors.prompts_for_plan`, `anchors.add_contained_negatives`, `sam_session.run_session` | the regression check (CONTRIBUTING.md); `tests/test_sam_session.py` |
| change the merge | `merge.read_candidates` (the readings), `merge.decide_pixels` (the handover), `merge.tiebreak_pick` (the dispute rule), `merge.merge_frame` | `docs/PIPELINE.md` Backward pass; `tests/test_merge.py`; the reproduction on a known clip |
| add a refusal reason | `records.REASON_CODES`, `schema/mask_record.schema.json` | the table in `docs/OUTPUT_FORMAT.md`; `tests/test_records.py` |
| add a pre-check | `worker.precheck_clip`, `video.decode_check` | a reason code as above; `tests/test_worker.py` |
| change what `export` writes | `export.MASK_SCHEMA` / `REFUSAL_SCHEMA`, `export_campaign` | `docs/OUTPUT_FORMAT.md`; `tests/test_export.py` |
| change the ConCor record | `concor.relation_record`, `tracklet_from_records`, the `*_SCHEMA` constants | `docs/CONCOR_VIDEO.md`; `tests/test_concor.py` |
| change how a campaign resumes or retries | `worker.unit_status`, `worker._run_unit`, `worker.process` | `docs/PIPELINE.md` Worker, `docs/SLURM.md`; `tests/test_worker.py` |
| support a new cluster | `scripts/submit_slurm.sh`, `slurm/process_array.slurm` only; no site path is hard-coded | `docs/SLURM.md` |
| pin a new sam3 commit or checkpoint | `__init__.py` (`SAM3_COMMIT`, `CHECKPOINT_SHA256`), `scripts/setup.sh`, `scripts/download_model.sh`, `sam3_compat.py` if the patches no longer apply | the regression check |

## The tests

`pytest -q` runs 124 tests in a few seconds on a CPU. `tests/conftest.py` builds a synthetic
dataset in a temporary directory (four videos with the shapes of the real files: human and
tracker boxes, a clip with 17 objects, a clip whose video is missing) and a 12-frame mp4; no
real data or model is touched. `tests/test_worker.py` drives `process` with a fake clip runner
that writes records without a GPU.

| file | tests | covers |
|---|---|---|
| `test_datasets.py` | 8 | the roots from the environment and the flags, loading all three splits, the join, the known counts, the depth-tolerant index, `resolve_video` preferring the transcode |
| `test_anchors.py` | 12 | spans, the reference anchor, thinning, contained negatives, prompt packing |
| `test_anchor_quality.py` | 9 | the gate flags, the fallback, the gap rule |
| `test_video.py` | 5 | decode checks, the black probe, the transcode (odd sizes kept) |
| `test_worker.py` | 23 | worklist and its count assertion, claims (exclusive, stale, concurrent), shard order, resume, status transitions, OOM retry, interruption, `--roots-from-env`, the subprocess command, `run_clip` with gap fills, negatives, both directions |
| `test_merge.py` | 13 | the readings, the pixels rule, the handover, the three dispute rules and both winners, the settings check |
| `test_records.py` | 22 (8 functions, one over every provenance field) | provenance, RLE round trip, atomic writes, validation |
| `test_sam_session.py` | 8 | the request sequence of a session against a stub predictor, closing on error, the score merge, the offload rule, clicks riding with the box corners, the prompt cap, the backward request |
| `test_export.py` | 4 | tables, ledger, manifest; missing and duplicate records, corrupt provenance, `--no-integrity` with pending clips |
| `test_concor.py` | 4 | the record, the validator, the tables |
| `test_cli.py` | 14 | every command through `main`, including `transcode --campaign-root` re-queueing and `render --side-by-side` |
| `test_no_torch.py` | 2 | the CPU modules import without torch |

## What is pinned

- The sam3 commit (`96914d2`) and the checkpoint sha256 (`0567debe...`): `__init__.py`,
  `scripts/setup.sh`, `scripts/download_model.sh`. `doctor` checks both; `process` asserts the
  hash before its first clip.
- The SAM path: eager mode, no FlashAttention, the request sequence of `run_session`. The masks
  regenerate byte-for-byte against earlier runs as long as these hold.
- The worklist format (`worker.WORKLIST_VERSION`) and the record contract (`records.py`, the
  schema). A record from any version of this package validates against the schema of that
  version; `code_commit` says which.
