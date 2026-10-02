# vidstg-masks

Produces one SAM 3.1 mask per (relation object, frame) for VidSTG: for every `used_relation`
in the VidSTG annotations, both the subject and the object trajectory are segmented over the
relation's frames, prompted with the VidOR ground-truth boxes at human-annotated keyframes.
Every record carries provenance (model, checkpoint hash, prompt, box origin, split, code
commit, timestamp). Where the input is doubtful the pipeline writes a refusal record with a
reason code instead of a mask.

Inputs: VidOR videos, VidOR annotation JSONs, VidSTG annotation JSONs. Outputs: video ids
and annotations only; no video content is copied into any output.

## Requirements

- Linux, Python 3.11+, ffmpeg and ffprobe on PATH.
- One NVIDIA GPU with 24 GB or more for `process` (measured on RTX A5000 24 GB). Every
  other command, and the tests, run on a CPU.
- Access to the gated `facebook/sam3.1` checkpoint on Hugging Face (accept the gate, then
  `hf auth login`).

## Data layout

```
$VIDOR_ANN_ROOT/**/<vid>.json                                 VidOR annotations (any folder depth)
$VIDOR_VIDEO_ROOT/<folder>/<vid>.mp4                          VidOR videos (annotation's video_path)
$VIDOR_TRANSCODED_ROOT/<folder>/<vid>.mp4                     optional H.264 re-encodes (see Limits)
$VIDSTG_ROOT/annotations/{train,val,test}_annotations.json    VidSTG annotations
```

The roots come from these environment variables or from `--*-root` flags; `.env.example`
lists every variable and `examples/*.env` hold complete sets for the three run types.
`build-worklist` freezes the roots and each clip's paths into `worklist.json`; if the compute
nodes mount the data elsewhere, run `process` with `--roots-from-env` (`docs/PIPELINE.md`).

## First run

Everything up to `process` needs no GPU and takes a few minutes on a login node.

```bash
git clone <this repository> vidstg-masks && cd vidstg-masks
INSTALL_GPU=0 bash scripts/setup.sh           # .venv + package + test extras, no torch
source .venv/bin/activate
pytest -q                                     # expect: 99 passed
cp .env.example .env && $EDITOR .env && set -a && source .env && set +a
vidstg-masks doctor --skip-hash --skip-gpu-libs     # expect: every "known fact" OK, "all checks passed"
vidstg-masks build-worklist --split val --campaign-root outputs/smoke --vids 7639717122
vidstg-masks plan --worklist outputs/smoke/worklist.json --vids 7639717122      # writes nothing
```

`doctor` checks the roots, ffmpeg, the sam3 commit, the checkpoint hash and the dataset
counts (44,808 VidSTG records, 6,770 videos, 26,016 relation pairs, 7,835 VidOR annotation
files); a mismatch means the annotation set is incomplete or altered. `build-worklist`
asserts the counts again before writing anything (`--no-assert-counts` for a partial copy).
Then, on a GPU node:

```bash
bash scripts/setup.sh                         # adds torch 2.11 cu128 and the pinned sam3 checkout
hf auth login && bash scripts/download_model.sh
vidstg-masks doctor --checkpoint "$SAM31_CHECKPOINT" --sam3-repo "$SAM31_REPO_ROOT"   # all checks passed
vidstg-masks process --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke \
    --checkpoint "$SAM31_CHECKPOINT" --shard-index 0 --shard-count 1
vidstg-masks status --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke
vidstg-masks export --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke
vidstg-masks export-concor --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke
vidstg-masks render --vid 7639717122 --campaign-root outputs/smoke
```

Expected on 7639717122 with the default policy: `507 masks / 0 refused`, about 0.5 s per frame
after a model load of about 1.5 min, 8 GB of VRAM; `export` prints `"ok": true` under
`integrity`; `render` writes `outputs/smoke/overlays/7639717122.mp4`; `export-concor` writes
`outputs/smoke/export/concor/tracklets.parquet` (18 rows: two per relation). These are the
numbers of the run that was checked against the research pipeline's masks (507 of 507
identical).

`setup.sh` and `download_model.sh` keep pip and Hugging Face caches under
`VIDSTG_MASKS_CACHE_ROOT` (default `<repo>/.cache`), never in `$HOME`, which is often
quota-limited on shared clusters. The token from `hf auth login` stays where the CLI keeps it.

## Whole splits

```bash
SPLIT=val CAMPAIGN_ROOT=outputs/val bash scripts/run_local.sh          # one machine, one GPU

SPLIT=all CAMPAIGN_ROOT=/outputs/vidstg-masks-all \
SLURM_PARTITION=gpu SLURM_ACCOUNT=myaccount GPU_GRES=gpu:1 NUM_WORKERS=16 \
bash scripts/submit_slurm.sh                                           # N GPUs, one array task each
```

`SPLIT` is `train`, `val`, `test` or `all`; `VIDS`, `VIDS_FILE` and `LIMIT` restrict the
worklist. Both scripts run `doctor` and `build-worklist` first, so a wrong dataset stops
before any GPU work. The array tasks share one worklist and claim clips with lease files, so
any `NUM_WORKERS` works; a task killed at its time limit requeues itself and the clip it was
running is retried. Re-running either script with the same `CAMPAIGN_ROOT` resumes: done clips
are skipped. `vidstg-masks status --worklist ... --campaign-root ...` is the progress view.
Details, the job graph and sizing: `docs/SLURM.md`; the done / failed / pending rules:
`docs/PIPELINE.md`.

Export writes `export/masks.parquet`, `export/refusals.parquet`, `export/ledger.csv` and
`export/manifest.json`, and checks that every boxed frame inside each relation span has
exactly one record, every RLE decodes, and every provenance field is set (`docs/OUTPUT_FORMAT.md`).

## Anchor policies

`--anchor-policy human_gap` (default): each object is prompted with its VidOR box at up to 16
human-annotated keyframes; the tracker boxes (about 97% of VidOR boxes) are never shown to
SAM. A coverage rule then puts keyframes back: wherever two consecutive anchors (or the span
start / end and the nearest anchor) are more than `--max-gap` frames apart (default 60; 0
disables the rule), the human keyframe nearest the middle of the gap is added, or with
`--gap-fill any` the tracker box nearest the middle where no human keyframe lies in the gap.
No quality gate; no video is needed for the plan. The fills are recorded in
`prompt_payload.gap`; `prompt_mode` is `pvs_box_multianchor`.

Contained negatives are on by default (any policy): overlap-aware prompts from the VidOR
boxes alone. Where another relation object's box lies inside an object's box and is small
relative to it (at least 90% inside, at most 25% of the area: a toy in a lap, a cup in a
hand), the container gets a negative click at the contained object's centre at each of its
anchors, and is prompted with its own box plus that click at the contained object's anchors
too (a tracker box where no human keyframe is there). Without them the larger object tends to
annex the smaller one between the smaller one's prompts. Recorded in
`prompt_payload.contained_negatives`; the added frames appear in `anchor_fids`.
`--no-contained-negatives` gives the plain box prompts.

`--anchor-policy human`: the keyframes alone, without the coverage rule — the default before
human_gap. `--anchor-policy human --no-contained-negatives` together reproduce that earlier
default: the same prompts as before.

`--anchor-policy hq`: the keyframes filtered by a quality gate (frame-edge contact, overlap
with other objects, motion, relative size, blur); an object with no clean keyframe keeps its
least-flagged ones, or is refused under `--hq-fallback refuse`; the coverage rule
(`--max-gap` / `--gap-fill`, as under human_gap) then puts keyframes back, and
`--keep-span-edges` (hq only) always anchors the first and last human keyframe of the span.
The gate's decisions and the fills are recorded in every record (`prompt_payload.hq`;
`prompt_mode` `pvs_box_hqanchor`). Thresholds were set on 16 VidSTG-val clips and have not
been validated at scale. Details and a glossary of terms: `docs/PIPELINE.md`.

Every `runs/<vid>.json` records `anchor_policy`, `max_anchors`, `hq_fallback`, `max_gap`,
`gap_fill`, `keep_span_edges` and `contained_negatives`, and `vidstg-masks status` lists
them, so a campaign's settings can be read off its outputs. These defaults are the command
line's; the library functions (`worker.process`, `anchors.plan_clip`, ...) take every setting
as an explicit argument and their own defaults are still the earlier ones, so a direct caller
should pass the settings it wants.

What was measured (20 VidSTG-val clips, 69,729 object-frames, 2026-09-30 and 2026-10-01; the
research repo's `reports/gap_rule.md`, numbers from its `src/report_gap_rule.py`): frames on
which SAM returned no mask, same clips, same frames, one run per arm:

| setting | frames without a mask | degenerate masks (of ~7,000 sampled) | masks mostly outside their own box |
|---|---:|---:|---:|
| `human --no-contained-negatives` (the earlier default) | 343 | 45 | 60 |
| `hq` alone | 2,480 | 42 | 58 |
| `hq` with the coverage rule | 511 | 62 | 57 |
| `human_gap --no-contained-negatives` | 309 | 50 | 58 |
| `hq --keep-span-edges` with contained negatives | 246 | 38 | 53 |
| `human_gap` with contained negatives (the default) | 137 | 31 | 50 |

The gate's flags remove the keyframes SAM needs most (fast motion, occlusion, entering and
leaving the frame), the coverage rule removes every loss in gaps longer than 60 frames, and the
negative clicks fix the cases where a large object takes a small object's pixels. The default
costs about four more prompts per object than the earlier default (18.1 against 14.4 anchors
per object, 531 negative clicks on 33 container objects over the 20 clips). The remaining 137
frames all lie between anchors fewer than 60 frames apart: occlusions and frames where the
object is leaving the picture, where the VidOR box is itself a tracker interpolation.

## Cost

Measured on one RTX A5000 (24 GB), eager mode, no FlashAttention 3:

| item | value |
|---|---|
| propagation | about 0.4 s per frame per object (0.45-0.8 s per frame at 5-8 objects) |
| model load | about 1.5 min per clip (each clip runs in a fresh process) |
| 451 val clips (rung-2 style selection) | about 92 GPU-h |
| all 602 val videos | about 120 GPU-h |
| all 6,770 videos | about 1,250-3,100 GPU-h depending on object counts |
| clips with frames x objects > 2,500 | run with tracker state on the CPU, 10-15% slower |
| peak VRAM | 6-19 GB per clip |


## Limits

- No verification and no captions: outputs are model masks with provenance, not reviewed
  masks. Multiplex masks of one clip are disjoint by construction.
- Clips with more than 16 relation objects are refused (`too_many_objects`, 9 of 6,770).
- About 2.8% of VidOR videos (VP6F codec, some with odd heights) decode as black frames
  under OpenCV, which is what SAM's loader uses. `process` probes every clip and refuses
  those with `decode_black`; `vidstg-masks transcode --worklist ... --campaign-root ...`
  re-encodes them to H.264 into `VIDOR_TRANSCODED_ROOT` (frame count asserted equal), after
  which re-running `process` picks them up.
- A decoded frame count or size that disagrees with the annotation refuses the clip
  (`frame_count_mismatch`, `frame_size_mismatch`); indices are never rescaled.
- An object with no human keyframe inside its span is refused (`no_human_keyframe`); a
  frame where SAM returns no pixels is refused (`empty_mask`).
- `mask_confidence` is the sigmoid of SAM's per-frame tracker score (1.0 on prompted
  frames). It is monotone but not calibrated.


## ConCor Video

The masks feed ConCor-Video-Data-Processing, so the package also writes its record contract:
`vidstg-masks export-concor --worklist ... --campaign-root ... [--captions captions.jsonl]`
gives one `concor-video-tracklet-bcc-v2` record per relation (two tracklets, uncompressed
COCO RLE aligned to the relation's frames, groups and span links from the caption) and their
`samples` / `tracklets` / `links` / `verification` Parquet tables, each record checked by the
same rules their validator applies. Relations without a caption go to `tracklets.parquet`
only. `scripts/run_local.sh` and the Slurm export job run it after `export`; set
`CAPTIONS=captions.jsonl` to add the full records. Mapping and open points:
`docs/CONCOR_VIDEO.md`.

## Documentation

- `docs/PIPELINE.md` — worklist, planning, pre-checks, the SAM session, records, the worker's
  retry rules, export, anchor policies, glossary.
- `docs/OUTPUT_FORMAT.md` — the record fields, reason codes, export tables.
- `docs/DATASETS.md` — what is read from VidSTG and VidOR and the counts asserted.
- `docs/SLURM.md` — the full-corpus command, job graph, walltime behaviour, resuming, sizing.
- `CONTRIBUTING.md` — tests to add with a change; what never goes into the repository.

## Licenses

SAM 3.1 code and weights are under the SAM License (facebookresearch/sam3, LICENSE); the
Hugging Face gate terms apply to the checkpoint, and publications must acknowledge SAM 3.
VidOR and VidSTG annotations and videos keep their own terms. Never redistribute the videos:
everything this pipeline writes contains video ids, frame indices, masks and provenance only.
The license for the code in this repository has not been chosen yet (Nathan decides before it is published).
