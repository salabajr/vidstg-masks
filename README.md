# vidstg-masks

Generate per-frame object masks for VidSTG relations using SAM 3.1, prompted
with VidOR ground-truth boxes.

For each relation, the pipeline tracks its subject and object across the
relation's frame interval and writes a mask (or an explicit refusal) for every
object-frame. Outputs include COCO RLE masks, provenance, refusal reasons,
and ConCor Video tables.

```
VidSTG annotations  (relations: subject, predicate, object, frame interval)  ─┐
VidOR annotations   (boxes per frame; human keyframes marked)                 ├─►  vidstg-masks  ─►  masks.parquet, refusals.parquet
VidOR videos                                                                 ─┘                     one record per object and frame,
                                                                                                    plus the ConCor Video tables
```

> **Known limitations.** Outputs are model-generated, not human-verified. Clips with more
> than 16 relation objects are refused, some VidOR videos require transcoding, and very
> large `both` runs can exceed host memory. See [Limits](#limits) for details.

**Contents:** [Pipeline overview](#pipeline-overview) · [Design principles](#design-principles) ·
[Requirements](#requirements) · [Quick start](#quick-start) · [A whole split](#a-whole-split-or-everything) ·
[Commands](#commands) · [Settings](#settings-that-change-the-masks) ·
[What comes out](#what-comes-out) · [ConCor Video](#concor-video) ·
[Code structure](#code-structure) · [Cost](#cost) · [Limits](#limits) ·
[Documentation](#documentation)

## Pipeline overview

1. **Build the worklist.** One unit per video: all objects in its relations, the combined
   clip span, and the frozen annotation and video paths.

2. **Validate the inputs.** The video must decode to the expected frame count and resolution
   and must not decode as black. Clips with more than 16 relation objects are refused.
   Frame indices always use native decode order.

3. **Choose prompts.** For each object, prompt SAM with its VidOR box at human-annotated
   frames only; tracker boxes are never shown. Keyframes are initially thinned to at most 16
   across the span, starting from the one with the least overlap with other objects. The gap
   rule then adds human keyframes wherever consecutive prompts are more than 60 frames apart,
   so the final number of prompts can exceed 16 on long spans. When a large object's box
   encloses a smaller one, the large object gets a negative click at the smaller one's centre.

4. **Run SAM.** One SAM 3.1 Object Multiplex session per clip with all objects together, so
   masks cannot overlap within a pass. One pass forward from the span start and one backward
   from the span end (`--direction forward` skips the backward pass, at half the GPU time).

5. **Merge passes.** The two passes are merged frame by frame without consulting
   the original boxes. Specks are dropped, a mask that appears in only one pass is kept, and
   the forward mask is used when both agree. Disagreements are refused, or resolved using
   `mask_confidence` when the higher-scoring pass clears a threshold. Overlapping pixels go to the object with the
   stronger claim, so final masks never overlap.

6. **Write records.** One record per object and boxed frame: either a COCO RLE mask with its
   confidence and full provenance, or a refusal with a reason code.

7. **Export.** Parquet tables, an integrity check that every boxed frame has exactly one
   record, a ledger, a manifest, and the ConCor Video tables. `render` paints the masks on the
   video for inspection.

Step by step: [docs/PIPELINE.md](docs/PIPELINE.md).

## Design principles

- **No silent failures.** A frame that cannot produce a usable mask is recorded as a refusal
  with a reason code, never skipped or guessed.
- **Reproducible inputs.** The worklist freezes the dataset roots and every clip's paths,
  and dataset counts are asserted before anything is written.
- **Native frame indices.** Frames use native decode order; mismatched frame counts or sizes
  are refused.
- **Full provenance.** Records carry the model, checkpoint hash, prompt, box origin, split,
  code commit and time.
- **No box-based merge.** Boxes prompt SAM but are never used to decide between propagated masks.
- **Auditable outputs.** Every boxed frame has exactly one record, checked at export, and
  `render` provides a visual QA path.

## Requirements

- Linux, Python 3.11 or newer, ffmpeg and ffprobe on the PATH.
- One NVIDIA GPU with 24 GB or more for the segmentation step. Everything else runs on a CPU.
- Access to the gated `facebook/sam3.1` checkpoint on Hugging Face: accept the gate on the
  website, then `hf auth login`.
- The data: VidOR videos and annotations, VidSTG annotations
  ([docs/DATASETS.md](docs/DATASETS.md)).

## Quick start

Steps 1 to 3 run on any machine, including a cluster login node. Steps 4 to 6 need a GPU.
A longer walkthrough with expected output and troubleshooting:
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

### 1. Install

```bash
git clone https://github.com/salabajr/vidstg-masks.git && cd vidstg-masks
INSTALL_GPU=0 bash scripts/setup.sh           # virtual environment and package, no torch
source .venv/bin/activate
pytest -q                                     # 126 tests pass in a few seconds
```

The setup script keeps pip and Hugging Face caches inside the repository (or under
`VIDSTG_MASKS_CACHE_ROOT`), not in your home directory, which is often quota-limited on
a cluster.

### 2. Configure the data

```
$VIDOR_ANN_ROOT/**/<vid>.json                                 VidOR annotations (any folder depth)
$VIDOR_VIDEO_ROOT/<folder>/<vid>.mp4                          VidOR videos (the annotation's video_path)
$VIDOR_TRANSCODED_ROOT/<folder>/<vid>.mp4                     optional H.264 re-encodes (see Limits)
$VIDSTG_ROOT/annotations/{train,val,test}_annotations.json    VidSTG annotations
```

```bash
cp .env.example .env && $EDITOR .env          # your data roots (leave SAM31_CHECKPOINT commented for now)
set -a && source ./.env && set +a
vidstg-masks doctor --skip-hash --skip-gpu-libs
```

`doctor` stops on a wrong or incomplete dataset: it expects 44,808 VidSTG records, 6,770
videos, 26,016 relation objects and 7,835 VidOR annotation files. `examples/*.env` contains
complete settings for a smoke test, one split and the whole corpus.

### 3. Build a one-video worklist

```bash
vidstg-masks build-worklist --split val --campaign-root outputs/smoke --vids 7639717122
```

The worklist freezes the roots and every clip's paths. If the compute nodes mount the data
elsewhere, run `process` with `--roots-from-env`.

### 4. Run segmentation (GPU node)

```bash
bash scripts/setup.sh                         # adds torch and the pinned sam3 checkout
hf auth login && bash scripts/download_model.sh
source .venv/bin/activate && set -a && source ./.env && set +a
vidstg-masks process --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke \
    --checkpoint "$SAM31_CHECKPOINT" --shard-index 0 --shard-count 1
```

### 5. Export

```bash
vidstg-masks export --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke
vidstg-masks export-concor --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke
```

### 6. Inspect the result

```bash
vidstg-masks render --vid 7639717122 --campaign-root outputs/smoke --side-by-side
```

This clip has 9 relations and 5 objects. You should see 507 masks and no refusals, about
half a second per frame per pass once the model is loaded, and `export` reporting `"ok": true`.
`outputs/smoke/overlays/7639717122.mp4` shows the original frames on the left and the masks,
boxes and labels on the right.

## A whole split, or everything

```bash
SPLIT=val CAMPAIGN_ROOT=outputs/val bash scripts/run_local.sh          # one machine, one GPU

SPLIT=all CAMPAIGN_ROOT=/outputs/vidstg-masks-all \
SLURM_PARTITION=gpu SLURM_ACCOUNT=myaccount GPU_GRES=gpu:1 NUM_WORKERS=16 \
bash scripts/submit_slurm.sh                                           # N GPUs, one array task each
```

`SPLIT` is `train`, `val`, `test` or `all`; `VIDS`, `VIDS_FILE` and `LIMIT` narrow it down.
`DIRECTION=forward DISPUTE_RULE=refuse` drops the backward pass and the merge, at half the GPU
time ([Settings](#settings-that-change-the-masks)).
`SLURM_ACCOUNT` is passed only when set.

Each clip runs in a fresh GPU subprocess and writes its own `records/<vid>.jsonl` atomically.
Array tasks share one worklist and claim clips one at a time, so any number of workers can run.
A task killed at its time limit requeues itself and the clip is retried. Running the same
command again with the same `CAMPAIGN_ROOT` resumes: finished clips are skipped.

```bash
vidstg-masks status --worklist $CAMPAIGN_ROOT/worklist.json --campaign-root $CAMPAIGN_ROOT
```

The job graph, every variable, walltime behaviour and sizing:
[docs/SLURM.md](docs/SLURM.md).

## Commands

One executable, `vidstg-masks`, with these subcommands. Every flag, with examples and what each
command reads and writes: [docs/CLI.md](docs/CLI.md).

| command | what it does | GPU |
|---|---|---|
| `doctor` | checks the data roots, ffmpeg, Python modules, the sam3 commit, checkpoint hash and dataset counts | no |
| `build-worklist` | selects videos of a split (or a list of vids) and writes `<campaign>/worklist.json` | no |
| `plan` | prints the anchors, prompts and negative clicks one or more clips would get, without running SAM | no |
| `process` | runs one shard of the worklist: claims clips, runs each in a fresh subprocess, writes records | yes |
| `process-one` | (internal) one clip in the current process; `process` calls it | yes |
| `status` | shows clips done / failed / pending, masks, refusals by reason and GPU hours | no |
| `export` | converts `records/*.jsonl` to `masks.parquet`, `refusals.parquet`, `ledger.csv` and `manifest.json`, with the integrity check | no |
| `export-concor` | exports the same masks in the ConCor Video record format | no |
| `render` | creates a QA video with masks, boxes and labels, optionally next to the original | no |
| `transcode` | H.264 re-encodes videos that decode black and re-queues their refused clips | no |

## Settings that change the masks

The defaults are the measured pipeline; most users change nothing. These settings control
prompt selection and how forward/backward passes are merged. All are recorded in
`runs/<vid>.json`, with prompt and merge details also stored in each record's `prompt_payload`.

| flag (launcher variable) | default | what it does |
|---|---|---|
| `--anchor-policy` (`ANCHOR_POLICY`) | `human_gap` | which human keyframes prompt SAM: `human` (up to 16, spread over the span), `human_gap` (the same plus the gap rule), `hq` (a quality gate first; measured worse, kept for comparison) |
| `--max-anchors` (`MAX_ANCHORS`) | 16 | the cap before the gap rule |
| `--max-gap` (`MAX_GAP`) | 60 | the gap rule: no stretch of an object's span longer than this without an anchor; 0 disables it |
| `--gap-fill` (`GAP_FILL`) | `human` | what may fill a gap: a human keyframe only, or `any` (a tracker box where no keyframe lies in the gap) |
| `--contained-negatives` (`CONTAINED_NEGATIVES`) | on | negative clicks for an object whose box contains a small object's box |
| `--direction` (`DIRECTION`) | `both` | `both`: a pass from the span start and one from the span end, merged frame by frame. `forward`: the first pass alone, half the GPU time. `backward`: the second pass alone |
| `--agree-iou`, `--speck-floor`, `--speck-ratio` | 0.3, 20, 0.1 | merge thresholds: when two masks agree and what counts as a speck |
| `--dispute-rule` (`DISPUTE_RULE`) | `higher_score` | a disputed frame (two real masks that do not overlap): `refuse` writes no mask; `higher_score` writes the pass with the higher `mask_confidence`; `forward_score` writes the forward mask. Both score rules need the score to reach `--dispute-score` |
| `--dispute-score` (`DISPUTE_SCORE`) | 0.907 | threshold of a score-based dispute rule (not allowed with `refuse`) |
| `--dispute-winner` (`DISPUTE_WINNER`) | `strong` | how a tie-break mask meets its neighbours: `weak` yields shared pixels to a stronger neighbour; `strong` takes them |

On 20 VidSTG-val clips (69,729 object-frames), the default prompts cut frames without a mask
from 343 to 137 compared with plain box prompts. The backward pass and merge, including the
default dispute tie-break (`higher_score` at 0.907 with the strong winner: 14 of 16 firm
labels), are described in [docs/PIPELINE.md](docs/PIPELINE.md#backward-pass-and-the-merge---direction).
These defaults are the setting of the 99-video VidSTG-val run of 2026-10-07. The forward pass
alone is `--direction forward --dispute-rule refuse`.

Look at `render --side-by-side` before trusting a new set of clips: the threshold rests on
few labeled frames.

## What comes out

```text
<campaign>/
  worklist.json          the units, dataset roots and counts
  records/<vid>.jsonl    one record per object and frame: a mask or a refusal
  runs/<vid>.json        timing, VRAM, counts and run settings
  errors/<vid>.json      the last failure of an unfinished clip
  overlays/<vid>.mp4     QA videos from `render`; never part of a release
  export/
    masks.parquet        one row per mask: vid, tid, fid, COCO RLE, confidence, provenance
    refusals.parquet     one row per refused frame, with its reason
    ledger.csv           every clip: done, failed or pending
    manifest.json        counts and an integrity check (every boxed frame has exactly one record)
    concor/              the same masks in the ConCor Video format
```

Every field, every reason code and the integrity check:
[docs/OUTPUT_FORMAT.md](docs/OUTPUT_FORMAT.md).

## ConCor Video

The masks feed [ConCor-Video-Data-Processing](https://github.com/suryathecreator/ConCor-Video-Data-Processing),
so `export-concor` writes its record format: one `concor-video-tracklet-bcc-v2` record per
relation, with one tracklet per object and a mask on every frame of the relation
(uncompressed COCO RLE, null where there is none), plus its `samples`, `tracklets`, `links`
and `verification` tables.

A full record also needs the relation's caption and the character spans of each object in it.
Give them with `--captions captions.jsonl` (or `CAPTIONS=...` for the launchers); every record
is then checked with the same rules as their validator, and the `verification` table opens in
their browser verifier. Without captions you get the `tracklets.parquet` table only.
The field mapping and open points: [docs/CONCOR_VIDEO.md](docs/CONCOR_VIDEO.md).

## Code structure

The main pipeline lives under `src/vidstg_masks/`:

- `anchors.py` — prompt/keyframe planning
- `sam_session.py` — SAM 3.1 inference
- `merge.py` — forward/backward merge
- `records.py` — record and provenance contract
- `export.py` — dataset exports
- `render.py` — visual QA

See [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md) for the full module map.

## Cost

With the defaults (two passes) the val split (602 videos) is about **80 GPU-hours** of
segmentation on one RTX A5000, about 150 task-hours as allocated by Slurm, and the full corpus
(6,770 videos) about **900 GPU-hours**, about 1,700 allocated. `--direction forward` roughly
halves both.

Measured on 99 VidSTG-val videos with the defaults (eager mode, 179,933 video frames, 243,750
boxed object-frames): 22.8 GPU-hours inside the SAM sessions and 42 task-hours allocated to the
array, model loads, idle waiting and retries included. The split and corpus figures above apply
the measured per-frame time to their worklists.

| item | value, both passes |
|---|---|
| segmentation | about 0.24 s per video frame plus 0.06 s per object (a 3-object clip: about 0.4 s per frame) |
| model load | about 1.5 min per clip |
| peak VRAM | 6 to 18 GB per clip (17.8 GB on the 99 videos) |

The merge itself runs on the CPU in seconds.

Host memory of a `both` run grows with frames × objects (about 14 MB per frame-object with the
tracker state offloaded to the CPU); [docs/SLURM.md](docs/SLURM.md#sizing) covers node sizing.

## Limits

- The outputs are model masks with provenance, not human-verified masks.
- Clips with more than 16 relation objects are refused (9 of 6,770 videos).
- About 2.8 percent of VidOR videos decode as black frames under OpenCV, which SAM's loader
  uses. Those clips are refused with `decode_black`; `vidstg-masks transcode --worklist ...
  --campaign-root ...` re-encodes them to H.264 into `VIDOR_TRANSCODED_ROOT` without changing
  the frame size, checks the result against the annotation, moves the clip's refusal records to
  `<campaign>/superseded/`, and the next `process` runs the clip.
- A decoded frame count or size that disagrees with the annotation refuses the clip.
  Frame indices are never rescaled.
- An object without a human keyframe in its span is refused; a frame where SAM returns no
  pixels is a refusal with `empty_mask`.
- `mask_confidence` is SAM's per-frame object score, between 0 and 1. It orders frames
  sensibly but is not calibrated.
- A clip whose frames × objects is very large may not fit a `both` run: one clip of 2,640
  frames × 8 objects (about 21,000 frame-objects) was killed four times by a 515 GB node's
  memory limit and then ran out of the 24 GB of VRAM; clips up to about 15,000 frame-objects
  ran. Such a clip is retried up to five times and then shows as failed, with the error in
  `errors/<vid>.json`; a forward-only run fits.

## Documentation

- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md): install, point at the data, one clip, expected output and troubleshooting.
- [docs/CLI.md](docs/CLI.md): every command and flag, what it reads and writes, examples.
- [docs/PIPELINE.md](docs/PIPELINE.md): the pipeline step by step, anchor policies, backward pass and merge.
- [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md): modules, the path of a clip through them, where to change what, and tests.
- [docs/OUTPUT_FORMAT.md](docs/OUTPUT_FORMAT.md): every record field, reason code and export table.
- [docs/CONCOR_VIDEO.md](docs/CONCOR_VIDEO.md): ConCor Video export and open points.
- [docs/DATASETS.md](docs/DATASETS.md): what is read from VidSTG and VidOR, and the counts that are checked.
- [docs/SLURM.md](docs/SLURM.md): full-corpus runs, job graph, resuming and sizing.
- [CHANGELOG.md](CHANGELOG.md): what changed, by version.
- [CONTRIBUTING.md](CONTRIBUTING.md): tests, repository rules and regression checks.
