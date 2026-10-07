# Getting started

A walkthrough from a fresh clone to the masks of one clip, with the output you should see at
each step, then how to run more clips, and a troubleshooting table. The README has the short
version; [docs/CLI.md](CLI.md) has every flag.

## 0. What you need

| item | why | where |
|---|---|---|
| Linux, Python 3.11 or newer | the package | `python3 --version` |
| ffmpeg and ffprobe on the PATH | decode checks, transcodes, the QA videos | `which ffmpeg ffprobe` |
| the VidOR videos and annotations | the frames and the boxes | [docs/DATASETS.md](DATASETS.md) |
| the VidSTG annotations | the relations | [docs/DATASETS.md](DATASETS.md) |
| one NVIDIA GPU with 24 GB or more | the segmentation step only | `nvidia-smi` |
| access to `facebook/sam3.1` on Hugging Face | the checkpoint is gated | accept the gate on the website, then `hf auth login` |

Everything except `process` runs on a CPU, so the install, the checks and the worklist can be
done on a login node and the GPU step on a compute node.

## 1. Install (CPU only)

```bash
git clone https://github.com/salabajr/vidstg-masks.git && cd vidstg-masks
INSTALL_GPU=0 bash scripts/setup.sh
source .venv/bin/activate
pytest -q
```

`setup.sh` creates `.venv`, installs the package with its test extras and prints:

```
environment=/path/to/vidstg-masks/.venv
install_gpu=0 (no torch, no sam3; tests, doctor --skip-gpu-libs, export and status work)
```

`pytest -q` ends with `123 passed`. The tests use synthetic annotations and a tiny generated
video; they never read the dataset.

The pip and Hugging Face caches go under `<repo>/.cache` (or `VIDSTG_MASKS_CACHE_ROOT`), not
under `$HOME`: a full, quota-limited home directory is the usual first failure on a cluster.

## 2. Point at the data

```bash
cp .env.example .env
$EDITOR .env               # VIDSTG_ROOT, VIDOR_ANN_ROOT, VIDOR_VIDEO_ROOT at least
set -a && source .env && set +a
```

The three roots are required by every command that reads annotations. The layout expected
under them:

```
$VIDSTG_ROOT/annotations/{train,val,test}_annotations.json
$VIDOR_ANN_ROOT/**/<vid>.json             any folder depth; the release has training/ and validation/
$VIDOR_VIDEO_ROOT/<folder>/<vid>.mp4      the annotation's video_path
```

Every root can also be given as a flag (`--vidstg-root`, `--vidor-ann-root`,
`--vidor-video-root`, `--vidor-transcoded-root`), which overrides the variable.

## 3. Check the installation and the data

```bash
vidstg-masks doctor --skip-hash --skip-gpu-libs
```

```
vidstg-masks 0.1.0 · python 3.12.13 · /path/to/vidstg-masks/.venv/bin/python
[OK  ] vidstg_root: /data/vidstg
[OK  ] vidor_ann_root: /data/vidor/annotation
[OK  ] vidor_video_root: /data/vidor/video
[--  ] vidor_transcoded_root: not set (optional)
[OK  ] vidstg train: /data/vidstg/annotations/train_annotations.json
[OK  ] vidstg val: /data/vidstg/annotations/val_annotations.json
[OK  ] vidstg test: /data/vidstg/annotations/test_annotations.json
[OK  ] ffmpeg: /usr/bin/ffmpeg
[OK  ] ffprobe: /usr/bin/ffprobe
[OK  ] numpy: 1.26.4
[OK  ] cv2: 4.10.0
[OK  ] pycocotools
[OK  ] pyarrow: 25.0.0
[--  ] torch / sam3: import skipped
[OK  ] known fact vidstg_records: 44,808 (expected 44,808)
[OK  ] known fact vidstg_videos: 6,770 (expected 6,770)
[OK  ] known fact relation_pairs: 26,016 (expected 26,016)
[OK  ] known fact vidor_annotations: 7,835 (expected 7,835)
[OK  ] VidSTG videos with a VidOR annotation: 6,770 / 6,770
       records per split: {'train': 36202, 'val': 3996, 'test': 4610}
all checks passed
```

A `[FAIL]` line names what is wrong; the exit code is 1 and nothing else should be run until
it passes. `--skip-hash` skips the sha256 of the 3.5 GB checkpoint (it is checked when the
GPU step loads it anyway), `--skip-gpu-libs` skips importing torch and sam3 (not installed
yet). `doctor --vid <vid>` additionally decodes one video and compares its frame count and
size with the annotation.

## 4. Build a worklist

A **campaign** is a directory that holds everything about one run: the worklist, the records,
the export. Start with one clip:

```bash
vidstg-masks build-worklist --split val --campaign-root outputs/smoke --vids 7639717122
```

```
{
 "worklist": "outputs/smoke/worklist.json",
 "counts": {
  "by_split": {"val": 1},
  "frames": 507,
  "prop_frame_objects": 2535,
  "relation_tids": 5,
  "too_many_objects": 0,
  "units": 1,
  "video_missing": 0
 }
}
```

`prop_frame_objects` is frames times objects summed over the clips: the number that drives the
GPU time (about 0.4 s each per pass). `--split` is `train`, `val`, `test` or `all`; `--vids`
or `--vids-file` picks videos, `--limit` caps the count. The dataset counts are asserted
first; a partial copy of the dataset needs `--no-assert-counts`.

Before spending GPU time you can see what a clip will be prompted with:

```bash
vidstg-masks plan --worklist outputs/smoke/worklist.json --vids 7639717122
```

```
== 6047872014 · 640x640 @ 10.00 fps · 65 frames · split val · policy human_gap
   video: /data/vidor/video/0010/6047872014.mp4
   relations (4): 1 kiss 0 | 1 hug 0 | 0 hug 1 | 0 kiss 1
   segment union [0, 64] -> propagation span [0, 64] (~65 frames)
   tid 0 (adult): span [0, 64] · 6/6 anchors · ref fid 0 (max IoU vs others 0.000) · gap rule (max gap 60, fill human): +0 fills over baseline [0, 21, 43, 53, 58, 64]
   tid 1 (adult): span [9, 64] · 13/13 anchors · ref fid 9 (max IoU vs others 0.000) · gap rule (max gap 60, fill human): +0 fills over baseline [9, 12, 15, 18, 21, 26, 32, 37, 43, 48, 53, 58, 64]
   contained negatives: 0 negative clicks, 0 co-prompts (0 at tracker boxes)
```

(the example is another clip, with two objects). One line per object: its span, how many of
its human keyframes become anchors, the reference anchor and why, and what the gap rule added.

## 5. Run one clip on a GPU

On a node with a GPU:

```bash
bash scripts/setup.sh                        # adds torch (CUDA 12.8 wheels) and the pinned sam3 checkout
hf auth login                                # once; the account must have accepted the facebook/sam3.1 gate
bash scripts/download_model.sh               # checkpoints/sam3.1/sam3.1_multiplex.pt, sha256 verified
set -a && source .env && set +a
vidstg-masks process --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke \
    --checkpoint checkpoints/sam3.1/sam3.1_multiplex.pt --shard-index 0 --shard-count 1
```

The worker prints one block per clip:

```
[worker] shard 0/1 owner=local:node1:12345:0 units=1 anchor_policy=human_gap max_anchors=16 ... direction=forward ...
[worker] checkpoint checkpoints/sam3.1/sam3.1_multiplex.pt: sha256 0567debeec80ba4a… == pinned
[clip] 6047872014 tids=2 frames=65 attempt=1
== 6047872014 · 640x640 @ 10.00 fps · 65 frames · split val · policy human_gap
   ... the plan, as printed by `plan` ...
   state offload off (65 frames x 2 objects)
   121 masks / 0 refused · 392 ms/f · 5.9 GB
[committed] 6047872014 rc=0 72s
[worker] shard 0: done=1 failed=0 pending=0
```

The first clip takes about 1.5 minutes longer than the rest: the model is loaded in every
per-clip subprocess. `ms/f` is the session time (prompts and propagation) per frame, `GB` the peak VRAM; `72s` is the whole subprocess, model load included.
With `--direction both` a `merge:` line counts what the merge did on the clip.

## 6. Look at what came out

```
outputs/smoke/
  worklist.json
  records/6047872014.jsonl      one line per object and frame
  runs/6047872014.json          timing, VRAM, counts, the settings
  claims/                       empty once the worker has finished
```

One record (abridged):

```json
{"vid": "6047872014", "tid": 0, "fid": 0,
 "rle": {"size": [640, 640], "counts": "\\R`599<Ub0Q1D>@b0_O=E..."},
 "model": "sam3.1-object-multiplex", "checkpoint_hash": "0567debe...", "sam_version": "3.1",
 "prompt_mode": "pvs_box_multianchor",
 "prompt_payload": {"anchor_fids": [0, 21, 43, 53, 58, 64], "ref_anchor_fid": 0, "anchor_policy": "human_gap", ...},
 "mask_confidence": 1.0, "box_generated": 0, "box_tracker": "none",
 "split": "val", "code_commit": "ef37f84", "created_at": "2026-10-07T02:17:18+00:00"}
```

A refusal has `"rle": null`, `"mask_confidence": null` and a `"reason_code"`. Every field:
[docs/OUTPUT_FORMAT.md](OUTPUT_FORMAT.md).

Progress and totals of a campaign, at any time and without writing anything:

```bash
vidstg-masks status --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke
```

```
campaign /path/outputs/smoke · worklist outputs/smoke/worklist.json · 1 clips
  done 1 · failed 0 · pending 0
  masks 121 · refusals 0 · GPU wall-hours 0.014
  run settings: anchor_policy=human_gap max_anchors=16 ... direction=forward ...
```

The tables, with the integrity check:

```bash
vidstg-masks export --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke
vidstg-masks export-concor --worklist outputs/smoke/worklist.json --campaign-root outputs/smoke
```

`export` prints its manifest; `"integrity": {"ok": true, ...}` means every boxed frame of
every finished clip has exactly one record, every mask decodes to the frame size and every
record validates against the schema. The exit code is 1 otherwise. `export-concor` writes the
ConCor Video tables next to them ([docs/CONCOR_VIDEO.md](CONCOR_VIDEO.md)).

And the video, for your eyes:

```bash
vidstg-masks render --vid 6047872014 --campaign-root outputs/smoke --side-by-side
```

writes `outputs/smoke/overlays/6047872014.mp4`: the original frames on the left, on the right
the masks tinted per object, each object's VidOR box (thick at a human keyframe, thin at a
tracker box), a label `tid:category` at each box, `(no mask)` after it where the object has a
box but no mask, and a banner with the relations. Without `--side-by-side` only the painted
frame is written.

## 7. More clips

A list of videos, one id per line (`#` comments allowed):

```bash
vidstg-masks build-worklist --split val --campaign-root outputs/val100 --vids-file my_vids.txt
vidstg-masks process --worklist outputs/val100/worklist.json --campaign-root outputs/val100 \
    --checkpoint checkpoints/sam3.1/sam3.1_multiplex.pt --shard-index 0 --shard-count 1
```

Several GPUs on one machine: one `process` per GPU with `--shard-index i --shard-count n` and
`CUDA_VISIBLE_DEVICES=i`; the shards share the worklist and claim clips one at a time.

A split or the whole corpus, end to end (checks, worklist, segmentation, both exports):

```bash
set -a; source examples/val.env; set +a; bash scripts/run_local.sh      # one GPU
set -a; source examples/val.env; set +a; bash scripts/submit_slurm.sh   # a Slurm array
```

Copy the example file and edit the paths; every variable is explained in it and in
[docs/SLURM.md](SLURM.md). Running the same command again on the same `CAMPAIGN_ROOT`
resumes: finished clips are skipped, interrupted ones retried.

The backward pass and the measured dispute tie-break (twice the GPU time, fewer frames
without a mask):

```bash
DIRECTION=both DISPUTE_RULE=higher_score DISPUTE_SCORE=0.907 DISPUTE_WINNER=strong \
    bash scripts/run_local.sh
```

or the same four flags on `process`: `--direction both --dispute-rule higher_score
--dispute-score 0.907 --dispute-winner strong`.

## Troubleshooting

| symptom | cause | what to do |
|---|---|---|
| `doctor`: `[FAIL] vidstg_root ... (missing)` | the variable is unset or points elsewhere | `set -a && source .env && set +a`, or pass `--vidstg-root` |
| `doctor`: `known fact vidstg_records: 20 (expected 44,808)` | a partial or altered copy of the annotations | get the release files; for a deliberate subset, `build-worklist --no-assert-counts` |
| `doctor`: `sam3 commit ... != pinned` | the sam3 checkout moved | `bash scripts/setup.sh` checks out the pinned commit again |
| `download_model.sh`: 401 or 403 | the account has not accepted the gate, or is not logged in | accept the gate at huggingface.co/facebook/sam3.1, `hf auth login` |
| `download_model.sh`: `checkpoint sha256 mismatch` | a different or truncated file | the script removes it; download again |
| `process`: every clip fails with `FileNotFoundError` | the worklist was built on a host with other mount points | run `process --roots-from-env` (or `ROOTS_FROM_ENV=1` for the launchers) with the roots set for this host |
| a clip is refused `decode_black` | a VP6F video that OpenCV decodes black | set `VIDOR_TRANSCODED_ROOT`, run `vidstg-masks transcode --worklist ... --campaign-root ...`, then `process` again |
| a clip is refused `frame_count_mismatch` or `frame_size_mismatch` | the video file does not match its annotation | check the file; the pipeline never rescales indices |
| `[oom]` in the worker log | the clip did not fit in VRAM | the retry with `--force-offload` is automatic; a second OOM is a terminal failure |
| `errors/<vid>.json` with `"interrupted": true` | the subprocess was killed (walltime, preemption, the kernel's memory killer) | it is retried on the next pass, five times in all; spread the tasks over more nodes ([docs/SLURM.md](SLURM.md#sizing)) |
| `export` exits 1 | the integrity check found a problem | the manifest lists the first 200; a clip's records can be regenerated by deleting its `records/<vid>.jsonl` and `runs/<vid>.json` and running `process` again |
| `$HOME` is full | pip or Hugging Face caches landed there | set `VIDSTG_MASKS_CACHE_ROOT` to scratch and run `setup.sh` again |
| a clip stays `pending` with a `claims/<vid>.claim` file | a worker died holding the lease | the lease is ignored after 30 minutes; nothing to do |
