# vidstg-masks

Mask tracks for VidSTG relations. For every relation in the VidSTG annotations, the subject
and the object are segmented on every frame of the relation with SAM 3.1 Object Multiplex,
prompted with their VidOR ground-truth boxes. Each mask carries its provenance (model,
checkpoint, prompt, box origin, split, code commit, time). Where there is no mask, the
pipeline writes a refusal with a reason instead of guessing.

Inputs: the VidOR videos, the VidOR annotations and the VidSTG annotations. Outputs: masks
and refusals as Parquet tables, plus the same masks in the record format of
[ConCor-Video-Data-Processing](https://github.com/suryathecreator/ConCor-Video-Data-Processing).
No video content is written to any output.

## Requirements

- Linux, Python 3.11 or newer, ffmpeg and ffprobe on the PATH.
- One NVIDIA GPU with 24 GB or more for the segmentation step. Everything else runs on a CPU.
- Access to the gated `facebook/sam3.1` checkpoint on Hugging Face: accept the gate on the
  website, then `hf auth login`.

## Data layout

```
$VIDOR_ANN_ROOT/**/<vid>.json                                 VidOR annotations (any folder depth)
$VIDOR_VIDEO_ROOT/<folder>/<vid>.mp4                          VidOR videos (the annotation's video_path)
$VIDOR_TRANSCODED_ROOT/<folder>/<vid>.mp4                     optional H.264 re-encodes (see Limits)
$VIDSTG_ROOT/annotations/{train,val,test}_annotations.json    VidSTG annotations
```

The roots come from environment variables or `--*-root` flags. `.env.example` lists every
variable; `examples/*.env` are complete settings for a smoke test, one split and the whole
corpus. The worklist freezes the roots and every clip's paths; if the compute nodes mount
the data elsewhere, run `process` with `--roots-from-env`.

## First run

Steps 1 to 4 need no GPU and take a few minutes on a login node.

```bash
git clone https://github.com/salabajr/vidstg-masks.git && cd vidstg-masks
INSTALL_GPU=0 bash scripts/setup.sh           # 1. virtual environment and the package, no torch
source .venv/bin/activate
pytest -q                                     # 2. 103 tests pass
cp .env.example .env && $EDITOR .env          # 3. your data roots
set -a && source .env && set +a
vidstg-masks doctor --skip-hash --skip-gpu-libs           # 4. checks the roots, ffmpeg, dataset counts
vidstg-masks build-worklist --split val --campaign-root outputs/smoke --vids 7639717122
```

`doctor` stops on a wrong or incomplete dataset: it expects 44,808 VidSTG records, 6,770
videos, 26,016 relation objects and 7,835 VidOR annotation files. Then, on a GPU node:

```bash
bash scripts/setup.sh                         # adds torch and the pinned sam3 checkout
hf auth login && bash scripts/download_model.sh
vidstg-masks process --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke \
    --checkpoint "$SAM31_CHECKPOINT" --shard-index 0 --shard-count 1
vidstg-masks export --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke
vidstg-masks export-concor --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke
vidstg-masks render --vid 7639717122 --campaign-root outputs/smoke     # overlay video, for your eyes
```

This clip has 9 relations and 5 objects. You should see 507 masks and no refusals, about
half a second per frame once the model is loaded, and `export` reporting `"ok": true`.

The setup script keeps pip and Hugging Face caches inside the repository (or under
`VIDSTG_MASKS_CACHE_ROOT`), not in your home directory, which is often quota-limited on a
cluster.

## A whole split, or everything

```bash
SPLIT=val CAMPAIGN_ROOT=outputs/val bash scripts/run_local.sh          # one machine, one GPU

SPLIT=all CAMPAIGN_ROOT=/outputs/vidstg-masks-all \
SLURM_PARTITION=gpu SLURM_ACCOUNT=myaccount GPU_GRES=gpu:1 NUM_WORKERS=16 \
bash scripts/submit_slurm.sh                                           # N GPUs, one array task each
```

`SPLIT` is `train`, `val`, `test` or `all`; `VIDS`, `VIDS_FILE` and `LIMIT` narrow it down.
Both scripts run the checks first, then the segmentation, then both exports. The array tasks
share one worklist and claim clips one at a time, so any number of workers works; a task
killed at its time limit requeues itself and the clip is retried. Running the same command
again with the same `CAMPAIGN_ROOT` resumes: finished clips are skipped. Progress:
`vidstg-masks status --worklist ... --campaign-root ...`. Details: `docs/SLURM.md`.

## What comes out

```
<campaign>/export/
  masks.parquet          one row per mask: vid, tid, fid, COCO RLE, confidence, provenance
  refusals.parquet       one row per refused frame, with its reason
  ledger.csv             every clip: done, failed or pending
  manifest.json          counts and an integrity check (every boxed frame has exactly one record)
  concor/                the same masks in the ConCor Video format (below)
```

Fields, reason codes and the integrity check: `docs/OUTPUT_FORMAT.md`.

## ConCor Video

The masks feed ConCor-Video-Data-Processing, so `export-concor` writes its record format:
one `concor-video-tracklet-bcc-v2` record per relation, with one tracklet per object and a
mask on every frame of the relation (uncompressed COCO RLE, null where there is none), plus
its `samples`, `tracklets`, `links` and `verification` tables.

A full record also needs the relation's caption and the character spans of each object in
it. Give them with `--captions captions.jsonl` (or `CAPTIONS=...` for the launchers); every
record is then checked with the same rules as their validator, and the `verification`
table opens in their browser verifier. Without captions you get the `tracklets.parquet`
table only. The field mapping and the points still open with the ConCor Video side:
`docs/CONCOR_VIDEO.md`.

## How the prompts are chosen

By default each object is prompted with its VidOR box at up to 16 human-annotated
keyframes, spread over its span; tracker boxes are never shown to SAM. Two rules improve on
that. Where two anchors are more than 60 frames apart, human keyframes are put back until no
gap is longer, because SAM's memory loses objects over long gaps. Where a small object's box
lies inside a larger object's box, the larger object gets a negative click at the small
object's centre, so it does not take the small object's pixels. On 20 VidSTG-val clips
(69,729 object-frames) this cut the frames without a mask from 343 to 137, at about four more
prompts per object.

Flags: `--anchor-policy human_gap` (the default), `--max-gap 60`, `--contained-negatives`
(default on, `--no-contained-negatives` to switch off). `--anchor-policy human
--no-contained-negatives` gives the plain box prompts. A third policy, `hq`, filters
keyframes by a quality gate; it did worse and is kept for comparison. The full description,
the measurements and the per-record fields that record every choice: `docs/PIPELINE.md`.

## Backward pass

SAM remembers an object best just after a prompt, so a forward pass fails most often on the
frames just before the next prompt. `--direction both` runs the same prompts a second time
from the end of the clip and merges the two passes frame by frame: where only one pass has a
mask it is taken, where both agree the forward mask is kept, and where they differ the mask
with more of its pixels inside the VidOR box wins, flagged `disputed` when the two barely
overlap (`--refuse-disputed` refuses those frames instead). Every record says which rule
decided it. This doubles the GPU time. `--direction backward` runs the backward pass alone.

## Cost

Measured on one RTX A5000 (24 GB), eager mode:

| item | value |
|---|---|
| segmentation | about 0.4 s per frame per object |
| model load | about 1.5 min per clip (each clip runs in a fresh process) |
| the whole val split (602 videos) | about 120 GPU-hours |
| all 6,770 videos | about 1,250 to 3,100 GPU-hours, depending on object counts |
| peak VRAM | 6 to 19 GB per clip |

## Limits

- The outputs are model masks with provenance, not human-verified masks.
- Clips with more than 16 relation objects are refused (9 of 6,770 videos).
- About 2.8 percent of VidOR videos decode as black frames under OpenCV, which SAM's loader
  uses. Those clips are refused with `decode_black`; `vidstg-masks transcode` re-encodes them
  to H.264 into `VIDOR_TRANSCODED_ROOT`, after which `process` picks them up.
- A decoded frame count or size that disagrees with the annotation refuses the clip.
  Frame indices are never rescaled.
- An object without a human keyframe in its span is refused; a frame where SAM returns no
  pixels is a refusal with `empty_mask`.
- `mask_confidence` is SAM's per-frame object score, between 0 and 1. It orders frames
  sensibly but is not calibrated.

## Documentation

- `docs/PIPELINE.md`: how the pipeline works, step by step, and the anchor policies.
- `docs/OUTPUT_FORMAT.md`: every field of a record, the reason codes, the export tables.
- `docs/CONCOR_VIDEO.md`: the ConCor Video export and the open points.
- `docs/DATASETS.md`: what is read from VidSTG and VidOR, and the counts that are checked.
- `docs/SLURM.md`: the full-corpus run, the job graph, resuming, sizing.
