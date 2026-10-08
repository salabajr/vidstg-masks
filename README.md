# vidstg-masks

Mask tracks for VidSTG relations. For every relation in the VidSTG annotations, the subject
and the object are segmented on every frame of the relation with SAM 3.1 Object Multiplex,
prompted with their VidOR ground-truth boxes. Each mask carries its provenance (model,
checkpoint, prompt, box origin, split, code commit, time). Where there is no mask, the
pipeline writes a refusal with a reason instead of guessing.

```
VidSTG annotations  (relations: subject, predicate, object, frame interval)  ─┐
VidOR annotations   (boxes per frame; human keyframes marked)                 ├─►  vidstg-masks  ─►  masks.parquet, refusals.parquet
VidOR videos                                                                 ─┘                     one record per object and frame,
                                                                                                    plus the ConCor Video tables
```

No video content is written to any output. The outputs are model masks with provenance
(Silver), never mixed with human-verified masks.

**Contents:** [How it works](#how-it-works-in-short) · [Requirements](#requirements) ·
[Quick start](#quick-start) · [A whole split](#a-whole-split-or-everything) ·
[Commands](#commands) · [Settings](#settings-that-change-the-masks) ·
[What comes out](#what-comes-out) · [ConCor Video](#concor-video) ·
[Code structure](#code-structure) · [Cost](#cost) · [Limits](#limits) ·
[Documentation](#documentation)

## How it works, in short

1. **Worklist.** One unit per video: the objects of all its relations, the clip span (the
   hull of the relation segments), the frozen annotation and video paths. The dataset counts
   are asserted before anything is written.
2. **Pre-checks.** The video must decode to the annotation's frame count and size and must not
   decode black; a clip with more than 16 relation objects is refused. Frame indices are
   native decode order and are never rescaled.
3. **Prompts.** Each object gets its VidOR box at up to 16 human keyframes (tracker boxes are
   never shown to SAM). The first prompt is the keyframe where the object overlaps the others
   least. Where two prompts are more than 60 frames apart, keyframes are put back (the gap
   rule). A large object boxed around a small one gets a negative click at the small one's
   centre (contained negatives).
4. **Segmentation.** One SAM 3.1 Object Multiplex session per clip with all its objects, so
   the masks of a frame never overlap inside a pass. One forward pass from the span start;
   with `--direction both`, a second pass from the span end.
5. **Merge** (`both` only). Frame by frame, without reading a box: specks are dropped, a pass
   that is alone is taken, two agreeing masks give the forward one, two disagreeing masks are
   a dispute (refused, or decided by SAM's presence score with `--dispute-rule`), and pixels
   two objects share go to the object with the stronger claim. No two masks share a pixel.
6. **Records.** One record per object and frame with a VidOR box: the mask (COCO RLE) with its
   confidence and full provenance, or a refusal with a reason code. Each clip runs in a fresh
   GPU subprocess and lands as `records/<vid>.jsonl`; a campaign can be resumed by path.
7. **Export.** Parquet tables with an integrity check (every boxed frame has exactly one
   record), a ledger, a manifest, and the ConCor Video tables. `render` draws the masks on the
   video for your eyes.

Step by step, with every rule and measurement: [docs/PIPELINE.md](docs/PIPELINE.md).

## Requirements

- Linux, Python 3.11 or newer, ffmpeg and ffprobe on the PATH.
- One NVIDIA GPU with 24 GB or more for the segmentation step. Everything else runs on a CPU.
- Access to the gated `facebook/sam3.1` checkpoint on Hugging Face: accept the gate on the
  website, then `hf auth login`.
- The data: VidOR videos and annotations, VidSTG annotations ([docs/DATASETS.md](docs/DATASETS.md)).

## Quick start

A longer walkthrough with the expected output of every step and a troubleshooting table:
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

### Data layout

```
$VIDOR_ANN_ROOT/**/<vid>.json                                 VidOR annotations (any folder depth)
$VIDOR_VIDEO_ROOT/<folder>/<vid>.mp4                          VidOR videos (the annotation's video_path)
$VIDOR_TRANSCODED_ROOT/<folder>/<vid>.mp4                     optional H.264 re-encodes (see Limits)
$VIDSTG_ROOT/annotations/{train,val,test}_annotations.json    VidSTG annotations
```

The roots come from environment variables or `--*-root` flags. `.env.example` holds the data
roots and the main Slurm settings; `examples/*.env` are complete settings for a smoke test, one
split and the whole corpus; every launcher variable is in [docs/SLURM.md](docs/SLURM.md). The worklist freezes the roots and every clip's paths; if the compute nodes mount
the data elsewhere, run `process` with `--roots-from-env`.

### On a CPU (a login node is enough)

```bash
git clone https://github.com/salabajr/vidstg-masks.git && cd vidstg-masks
INSTALL_GPU=0 bash scripts/setup.sh           # 1. virtual environment and the package, no torch
source .venv/bin/activate
pytest -q                                     # 2. 126 tests pass in a few seconds
cp .env.example .env && $EDITOR .env          # 3. your data roots (leave SAM31_CHECKPOINT commented for now)
set -a && source ./.env && set +a
vidstg-masks doctor --skip-hash --skip-gpu-libs           # 4. checks the roots, ffmpeg, dataset counts
vidstg-masks build-worklist --split val --campaign-root outputs/smoke --vids 7639717122
```

`doctor` stops on a wrong or incomplete dataset: it expects 44,808 VidSTG records, 6,770
videos, 26,016 relation objects and 7,835 VidOR annotation files.

### On a GPU node

```bash
bash scripts/setup.sh                         # adds torch and the pinned sam3 checkout
hf auth login && bash scripts/download_model.sh
source .venv/bin/activate && set -a && source ./.env && set +a
vidstg-masks process --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke \
    --checkpoint "$SAM31_CHECKPOINT" --shard-index 0 --shard-count 1
vidstg-masks export --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke
vidstg-masks export-concor --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke
vidstg-masks render --vid 7639717122 --campaign-root outputs/smoke --side-by-side   # the video next to the masks
```

This clip has 9 relations and 5 objects. You should see 507 masks and no refusals, about
half a second per frame once the model is loaded, and `export` reporting `"ok": true`.
`outputs/smoke/overlays/7639717122.mp4` shows the original frames on the left and the masks,
boxes and labels on the right.

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
`DIRECTION=both DISPUTE_RULE=higher_score DISPUTE_SCORE=0.907 DISPUTE_WINNER=strong` adds the
backward pass and the measured dispute tie-break ([Settings](#settings-that-change-the-masks));
`SLURM_ACCOUNT` is passed only when set. Both scripts run the checks first, then the
segmentation, then both exports. The array tasks share one worklist and claim clips one at a
time, so any number of workers works; a task killed at its time limit requeues itself and the
clip is retried. Running the same command again with the same `CAMPAIGN_ROOT` resumes:
finished clips are skipped. Progress:

```bash
vidstg-masks status --worklist $CAMPAIGN_ROOT/worklist.json --campaign-root $CAMPAIGN_ROOT
```

The job graph, every variable, the walltime behaviour and sizing: [docs/SLURM.md](docs/SLURM.md).

## Commands

One executable, `vidstg-masks`, with these subcommands. Every flag, with examples and what each
command reads and writes: [docs/CLI.md](docs/CLI.md).

| command | what it does | GPU |
|---|---|---|
| `doctor` | checks the data roots, ffmpeg, the Python modules, the sam3 commit, the checkpoint hash and the dataset counts | no |
| `build-worklist` | selects the videos of a split (or a list of vids) and writes `<campaign>/worklist.json` | no |
| `plan` | prints the anchors, prompts and negative clicks one or more clips would get, without running SAM | no |
| `process` | runs one shard of the worklist: claims clips, runs each in a fresh subprocess, writes records | yes |
| `process-one` | (internal) one clip in the current process; `process` calls it | yes |
| `status` | progress of a campaign: clips done / failed / pending, masks, refusals by reason, GPU hours | no |
| `export` | `records/*.jsonl` to `masks.parquet`, `refusals.parquet`, `ledger.csv`, `manifest.json`, with the integrity check | no |
| `export-concor` | the same masks in the ConCor Video record format | no |
| `render` | a QA video of one clip: masks tinted per object, boxes, labels, optionally next to the original | no |
| `transcode` | H.264 re-encodes of the videos that decode black, and re-queues the clips they refused | no |

## Settings that change the masks

Everything else is bookkeeping. Each of these is recorded in every `runs/<vid>.json`, and the
records' `prompt_payload` carries the anchor policy, the gap fills, the negative clicks and, in a
`both` run, the merge settings, so a campaign's outputs say how it was run.

| flag (launcher variable) | default | what it does |
|---|---|---|
| `--anchor-policy` (`ANCHOR_POLICY`) | `human_gap` | which human keyframes prompt SAM: `human` (up to 16, spread over the span), `human_gap` (the same plus the gap rule), `hq` (a quality gate first; measured worse, kept for comparison) |
| `--max-anchors` (`MAX_ANCHORS`) | 16 | the cap before the gap rule |
| `--max-gap` (`MAX_GAP`) | 60 | the gap rule: no stretch of an object's span longer than this without an anchor; 0 disables it |
| `--gap-fill` (`GAP_FILL`) | `human` | what may fill a gap: a human keyframe only, or `any` (a tracker box where no keyframe lies in the gap) |
| `--contained-negatives` (`CONTAINED_NEGATIVES`) | on | negative clicks for an object whose box contains a small object's box |
| `--direction` (`DIRECTION`) | `forward` | `forward`: one pass from the span start. `both`: a second pass from the span end, merged frame by frame; twice the GPU time. `backward`: the second pass alone |
| `--agree-iou`, `--speck-floor`, `--speck-ratio` | 0.3, 20, 0.1 | the merge's thresholds: when two masks agree, what counts as a speck |
| `--dispute-rule` (`DISPUTE_RULE`) | `refuse` | a disputed frame (two real masks that do not overlap): `refuse` writes no mask; `higher_score` writes the pass with the higher presence score when it is at least `--dispute-score`; `forward_score` writes the forward mask when its score is at least `--dispute-score` |
| `--dispute-score` (`DISPUTE_SCORE`) | unset | the threshold of a score rule (required with one, not allowed with `refuse`) |
| `--dispute-winner` (`DISPUTE_WINNER`) | `weak` | how a tie-break mask meets its neighbours: `weak` yields the pixels it shares with a stronger neighbour, `strong` takes them |

On 20 VidSTG-val clips (69,729 object-frames) the default prompts cut the frames without a
mask from 343 to 137 compared with plain box prompts. The backward pass and the merge, and
the one tie-break that agreed with the reviewer's frame labels (`higher_score` at 0.907 with
the strong winner: 14 of 16 firm labels), are described with their counts and their cautions
in [docs/PIPELINE.md](docs/PIPELINE.md#backward-pass-and-the-merge---direction). The setting is

```
--direction both --dispute-rule higher_score --dispute-score 0.907 --dispute-winner strong
```

Look at a run's videos (`render --side-by-side`) before trusting a new set of clips: the
threshold rests on few labeled frames.

## What comes out

```
<campaign>/
  worklist.json          the units, the dataset roots, the counts
  records/<vid>.jsonl    one record per object and frame: a mask or a refusal
  runs/<vid>.json        timing, VRAM, counts and the run settings of a finished clip
  errors/<vid>.json      the last failure of a clip that is not finished
  overlays/<vid>.mp4     QA videos from `render`; never part of a release
  export/
    masks.parquet        one row per mask: vid, tid, fid, COCO RLE, confidence, provenance
    refusals.parquet     one row per refused frame, with its reason
    ledger.csv           every clip: done, failed or pending
    manifest.json        counts and an integrity check (every boxed frame has exactly one record)
    concor/              the same masks in the ConCor Video format (below)
```

Every field, every reason code and the integrity check: [docs/OUTPUT_FORMAT.md](docs/OUTPUT_FORMAT.md).

## ConCor Video

The masks feed [ConCor-Video-Data-Processing](https://github.com/suryathecreator/ConCor-Video-Data-Processing),
so `export-concor` writes its record format: one `concor-video-tracklet-bcc-v2` record per
relation, with one tracklet per object and a mask on every frame of the relation
(uncompressed COCO RLE, null where there is none), plus its `samples`, `tracklets`, `links`
and `verification` tables.

A full record also needs the relation's caption and the character spans of each object in
it. Give them with `--captions captions.jsonl` (or `CAPTIONS=...` for the launchers); every
record is then checked with the same rules as their validator, and the `verification`
table opens in their browser verifier. Without captions you get the `tracklets.parquet`
table only. The field mapping and the points still open with the ConCor Video side:
[docs/CONCOR_VIDEO.md](docs/CONCOR_VIDEO.md).

## Code structure

```
src/vidstg_masks/
  cli.py             the `vidstg-masks` command: argument parsing, one function per subcommand
  datasets.py        VidSTG and VidOR loading, the dataset roots, the known counts, video resolution
  anchors.py         the plan of a clip from the annotations alone: objects, spans, anchors, prompts, negative clicks
  anchor_quality.py  the gap rule and the optional quality gate over keyframes
  video.py           decode checks (frame count, size, black frames) and the H.264 transcode
  worker.py          worklist, claims, the shard loop, the per-clip subprocess, records of a clip, status
  sam_session.py     the SAM 3.1 session: load the checkpoint, prompt, propagate, collect scores
  sam3_compat.py     the patches the vendored sam3 commit needs for box-prompted video sessions
  merge.py           the merge of the forward and backward pass (the pixels rule, the dispute tie-break)
  records.py         the record contract: provenance fields, reason codes, RLE, atomic JSONL, validation
  export.py          Parquet tables, ledger, manifest, integrity check
  concor.py          the ConCor Video export
  render.py          the QA video
schema/mask_record.schema.json    the record contract as JSON Schema
scripts/             setup.sh, download_model.sh, run_local.sh, submit_slurm.sh
slurm/process_array.slurm         one array task = one GPU = one shard
tests/               126 CPU-only tests on synthetic annotations and a tiny video (pytest -q)
examples/            complete .env files for a smoke test, the val split and the whole corpus
docs/                the documentation listed below
```

What each module owns, how a clip travels through them, and where to change what:
[docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md).

## Cost

Measured on one RTX A5000 (24 GB), eager mode:

| item | value |
|---|---|
| segmentation | about 0.4 s per frame per object |
| model load | about 1.5 min per clip (each clip runs in a fresh process) |
| the whole val split (602 videos) | about 120 GPU-hours |
| all 6,770 videos | about 1,250 to 3,100 GPU-hours, depending on object counts |
| peak VRAM | 6 to 19 GB per clip |

These are the numbers of one pass. `--direction both` runs two passes, so the segmentation time
and the GPU-hours double; the merge itself runs on the CPU in seconds. Measured on 99 VidSTG-val
videos with `--direction both` (243,750 object-frames, peak VRAM 17.8 GB): 22.8 GPU-hours inside
the SAM sessions, 42 task-hours allocated to the array including idle waiting and retries. Host
memory of a `both` run grows with frames times objects (about 14 MB per frame-object with the
tracker state offloaded to the CPU); [docs/SLURM.md](docs/SLURM.md#sizing) says how to size the
nodes.

## Limits

- The outputs are model masks with provenance, not human-verified masks.
- Clips with more than 16 relation objects are refused (9 of 6,770 videos).
- About 2.8 percent of VidOR videos decode as black frames under OpenCV, which SAM's loader
  uses. Those clips are refused with `decode_black`; `vidstg-masks transcode --worklist ...
  --campaign-root ...` re-encodes them to H.264 into `VIDOR_TRANSCODED_ROOT` without changing
  the frame size (an odd width or height is written as 4:4:4 chroma), checks the result against
  the annotation, moves the clip's refusal records to `<campaign>/superseded/` and the next
  `process` runs the clip. Running `submit_slurm.sh` or `run_local.sh` again on the same campaign
  is that `process`.
- A decoded frame count or size that disagrees with the annotation refuses the clip.
  Frame indices are never rescaled.
- An object without a human keyframe in its span is refused; a frame where SAM returns no
  pixels is a refusal with `empty_mask`.
- `mask_confidence` is SAM's per-frame object score, between 0 and 1. It orders frames
  sensibly but is not calibrated.
- A clip whose frames times objects is very large may not fit a `both` run: one clip of 2,640
  frames × 8 objects (about 21,000 frame-objects) was killed four times by a 515 GB node's
  memory limit and then ran out of the 24 GB of VRAM; clips up to about 15,000 frame-objects
  ran. Such a clip is retried up to five times and then shows as failed, with the error in
  `errors/<vid>.json`; a forward-only run of it fits.

## Documentation

- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md): install, point at the data, one clip, the expected output of every step, troubleshooting.
- [docs/CLI.md](docs/CLI.md): every command and flag, what it reads and writes, examples.
- [docs/PIPELINE.md](docs/PIPELINE.md): how the pipeline works, step by step; the anchor policies; the backward pass and the merge.
- [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md): the modules, the path of one clip through them, where to change what, the tests.
- [docs/OUTPUT_FORMAT.md](docs/OUTPUT_FORMAT.md): every field of a record, the reason codes, the export tables.
- [docs/CONCOR_VIDEO.md](docs/CONCOR_VIDEO.md): the ConCor Video export and the open points.
- [docs/DATASETS.md](docs/DATASETS.md): what is read from VidSTG and VidOR, and the counts that are checked.
- [docs/SLURM.md](docs/SLURM.md): the full-corpus run, the job graph, resuming, sizing.
- [CHANGELOG.md](CHANGELOG.md): what changed, by version.
- [CONTRIBUTING.md](CONTRIBUTING.md): tests, what never goes into the repository, the regression check.
