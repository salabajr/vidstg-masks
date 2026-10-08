# Command line

One executable, `vidstg-masks`, with ten subcommands. `vidstg-masks <command> --help` prints
the flags; this page says what each command reads and writes and shows an example. All paths
may be relative to the current directory. Every command exits 0 on success; the exceptions
are noted.

Commands that read annotations take the **roots** group of flags, which override the
environment variables of the same name:

```
--vidstg-root DIR            VIDSTG_ROOT
--vidor-ann-root DIR         VIDOR_ANN_ROOT
--vidor-video-root DIR       VIDOR_VIDEO_ROOT
--vidor-transcoded-root DIR  VIDOR_TRANSCODED_ROOT (optional; searched before the video root)
```

Commands that select clips take `--vids VID [VID ...]` and `--vids-file FILE` (one id per
line, first token, `#` comments).

| command | GPU | reads | writes |
|---|---|---|---|
| [`doctor`](#doctor) | no | roots, annotations, checkpoint | nothing |
| [`build-worklist`](#build-worklist) | no | annotations | `<campaign>/worklist.json` |
| [`plan`](#plan) | no | worklist, annotations | nothing |
| [`process`](#process) | yes | worklist, annotations, videos, checkpoint | `records/`, `runs/`, `errors/`, `claims/` |
| [`process-one`](#process-one) | yes | the same | the same, one clip |
| [`status`](#status) | no | worklist, `runs/`, `records/`, `errors/` | nothing |
| [`export`](#export) | no | worklist, `records/`, annotations | `export/` |
| [`export-concor`](#export-concor) | no | worklist, `records/`, annotations, captions | `export/concor/` |
| [`render`](#render) | no | `records/`, annotations, video | `overlays/<vid>.mp4` |
| [`transcode`](#transcode) | no | videos, annotations, worklist, `runs/` | `$VIDOR_TRANSCODED_ROOT/`, `<campaign>/superseded/` |

## doctor

```
vidstg-masks doctor [roots] [--checkpoint FILE] [--sam3-repo DIR] [--skip-hash] [--skip-facts] [--skip-gpu-libs] [--vid VID]
```

Checks, one `[OK  ]` / `[FAIL]` / `[--  ]` line each: the four roots and the three VidSTG
files exist; ffmpeg and ffprobe are on the PATH; numpy, cv2, pycocotools and pyarrow import;
whether torch and sam3 import and see a GPU (reported, never a failure; `--skip-gpu-libs` skips
the import); the sam3 checkout is at the
pinned commit (`--sam3-repo`); the checkpoint exists and has the pinned sha256 (`--checkpoint`;
`--skip-hash` skips the 3.5 GB read); the annotation set carries the release counts (44,808
VidSTG records, 6,770 videos, 26,016 relation objects, 7,835 VidOR files; `--skip-facts`
skips loading them). `--vid` also decodes one video and compares its frame count and size
with the annotation and probes for black frames. Exit code 1 when any check fails.

```bash
vidstg-masks doctor --skip-hash --skip-gpu-libs                        # login node
vidstg-masks doctor --checkpoint "$SAM31_CHECKPOINT" --vid 7639717122   # GPU node, one video probed
```

## build-worklist

```
vidstg-masks build-worklist [roots] --split {train,val,test,all} --campaign-root DIR
                            [--vids ...] [--vids-file FILE] [--limit N] [--if-missing] [--no-assert-counts]
```

Selects the videos of the VidSTG file(s) named by `--split` (narrowed by `--vids`,
`--vids-file`, `--limit`), asserts the dataset counts (`--no-assert-counts` for a deliberate
subset of the data), and writes `<campaign>/worklist.json`: one unit per video with its
relations, segment (earliest `used_segment` start to latest end), object categories, `n_tids`, `n_frames`, and the absolute annotation
and video paths as resolved on this host; plus the roots and the counts (`units`, `by_split`,
`frames`, `prop_frame_objects`, `relation_tids`, `video_missing`, `too_many_objects`). Prints the
path and the counts as JSON. `--if-missing` keeps an existing worklist (the launchers use it to
resume). A worklist is never rewritten by `process`, except by `transcode --campaign-root`,
which points a repaired unit at its new video file.

```bash
vidstg-masks build-worklist --split val --campaign-root outputs/val
vidstg-masks build-worklist --split all --campaign-root outputs/sub --vids-file clips.txt --no-assert-counts
```

## plan

```
vidstg-masks plan --worklist FILE --vids VID [VID ...] [policy flags]
```

Prints, without touching a GPU, what `process` would prompt for each clip: the span, each
object's anchors (how many of its human keyframes, the reference anchor and its overlap with
the other objects, the gap rule's fills), the negative clicks and co-prompts. The policy flags
are those of `process` (`--anchor-policy`, `--max-anchors`, `--max-gap`, `--gap-fill`,
`--hq-fallback`, `--keep-span-edges`, `--contained-negatives`); the merge flags are accepted
for symmetry and ignored. Under `--anchor-policy hq` a one-line summary of the gate per object
(kept, dropped and why, fills) is printed too; this reads the video.

```bash
vidstg-masks plan --worklist outputs/val/worklist.json --vids 7639717122 2953174101
```

## process

```
vidstg-masks process [roots] --worklist FILE --campaign-root DIR --checkpoint FILE
                     [--shard-index I] [--shard-count N] [--roots-from-env] [--max-attempts N] [--python EXE]
                     [--anchor-policy {human,human_gap,hq}] [--max-anchors N] [--max-gap N] [--gap-fill {human,any}]
                     [--hq-fallback {least-flagged,refuse}] [--keep-span-edges] [--contained-negatives | --no-contained-negatives]
                     [--direction {forward,backward,both}] [--agree-iou X] [--speck-floor N] [--speck-ratio X]
                     [--dispute-rule {refuse,forward_score,higher_score}] [--dispute-score T] [--dispute-winner {weak,strong}]
```

One shard of a campaign on one GPU. The shard walks the whole worklist starting at its own
block (`--shard-index` of `--shard-count`), skips clips that are done, claims each pending clip
with an exclusive lease file (`claims/<vid>.claim`, heartbeat every 30 s, ignored after 30
minutes), and runs it in a fresh subprocess (`process-one`), because SAM 3.1 does not release
session memory between clips. A finished clip is `records/<vid>.jsonl` plus `runs/<vid>.json`;
a failed attempt is `errors/<vid>.json`. When everything left is claimed by other shards the
shard waits for them rather than leave a tail behind. The checkpoint's sha256 is computed once
per shard and asserted against the pinned value.

Clip-level refusals that need no GPU (`too_many_objects`, `video_missing`) are written by the
shard itself. The subprocess runs the pre-checks (frame count, size, black decode), the plan,
the SAM session(s) and the merge, and writes the records.

Retries: a CUDA out-of-memory is retried once with the tracker state offloaded to the CPU
(`--max-attempts`, default 2); a subprocess killed by a signal (walltime, preemption, the
kernel's memory killer) is recorded as interrupted and retried up to 5 times; any other error
is terminal after one attempt and shows as `failed`. On `SIGUSR1` or `SIGTERM` the shard stops
claiming, finishes the active clip if it can, and exits 99 (the Slurm script requeues it).

`--roots-from-env` replaces the roots frozen in the worklist with the environment variables
or `--*-root` flags and rebases every unit's paths onto them: for compute nodes that mount the
data elsewhere than the node that built the worklist. `--python` chooses the interpreter of
the subprocess (default: the current one).

The policy and merge flags are described in [README, Settings](../README.md#settings-that-change-the-masks)
and in [docs/PIPELINE.md](PIPELINE.md); `--dispute-score` defaults to 0.907 with a score rule and
is not allowed with `refuse` (the command stops with a message otherwise). The defaults are the
measured pipeline: `--direction both --dispute-rule higher_score --dispute-score 0.907
--dispute-winner strong`. Every setting is written
into each `runs/<vid>.json`.

Exit code 0 when no clip of this shard failed terminally, 1 otherwise (also on a bad shard index
or bad dispute settings, with a message), 99 when drained on a signal.

```bash
vidstg-masks process --worklist outputs/val/worklist.json --campaign-root outputs/val \
    --checkpoint "$SAM31_CHECKPOINT" --shard-index 0 --shard-count 4 \
    --direction forward --dispute-rule refuse        # the forward pass alone; omit for the defaults
```

## process-one

```
vidstg-masks process-one [roots] --worklist FILE --campaign-root DIR --checkpoint FILE --vid VID
                         [--attempt N] [--force-offload] [--checkpoint-hash SHA256] [policy and merge flags]
```

Internal: one clip in the current process, exactly as `process` runs it in a subprocess. Useful
to run or debug a single clip in the foreground. `--force-offload` puts the tracker state on
the CPU from the start (the OOM retry does this); `--checkpoint-hash` skips the sha256 when the
parent has verified it; `--attempt` is recorded in the error file on failure. Exit code 0 when
the clip committed, 3 on a CUDA out-of-memory, 2 when the vid is not in the worklist, 1 on any
other error (recorded in `errors/<vid>.json`).

```bash
vidstg-masks process-one --worklist outputs/val/worklist.json --campaign-root outputs/val \
    --checkpoint "$SAM31_CHECKPOINT" --vid 7639717122
```

## status

```
vidstg-masks status --worklist FILE --campaign-root DIR [--json]
```

Progress of a campaign from the files on disk, writing nothing: clips done / failed / pending
(with the number of interrupted clips awaiting retry), masks and refusals so far, refusals by
reason code, GPU wall-hours (the sum of the clips' SAM session times: prompts and propagation; the model load is not counted), the distinct run
settings found in `runs/*.json`, and the failed clips with their error type. `--json` prints
the same as one object. Runs on a login node while the workers run.

```bash
vidstg-masks status --worklist outputs/val/worklist.json --campaign-root outputs/val
```

```
campaign /path/outputs/val100 · worklist /path/outputs/val100/worklist.json · 100 clips
  done 99 · failed 0 · pending 1 (of which 1 interrupted, awaiting retry)
  masks 242,889 · refusals 861 · GPU wall-hours 22.828
  refusals by reason: disputed_mask 17, empty_mask 460, passes_conflict 80, speck_mask 304
  run settings: anchor_policy=human_gap max_anchors=16 ... direction=both ... dispute_rule=higher_score dispute_winner=strong
```

## export

```
vidstg-masks export --worklist FILE --campaign-root DIR [--output-dir DIR] [--no-integrity]
```

Streams `records/*.jsonl` of the done clips into `export/masks.parquet` and
`export/refusals.parquet` (zstd), writes `export/ledger.csv` (every unit: done, failed or
pending, with counts and timing) and `export/manifest.json` (counts, refusals by reason, code
commits, checkpoint hashes, GPU hours, the integrity result). The integrity check recomputes
each clip's boxed in-span (tid, fid) set from the VidOR annotation and requires exactly one
record for each, decodes every RLE to the frame size and validates every record against the
schema; problems are listed in the manifest (first 200) and make the command exit 1.
`--no-integrity` skips the check. Re-runnable at any time, also while workers are running
(the export covers what is done at that moment).

```bash
vidstg-masks export --worklist outputs/val/worklist.json --campaign-root outputs/val
```

## export-concor

```
vidstg-masks export-concor --worklist FILE --campaign-root DIR [--output-dir DIR] [--captions FILE]
```

Writes the same masks in the ConCor Video record format under `export/concor/`: one record
per relation (`records/<sample_id>.json` with `:` written as `_`, captioned relations only, each
validated with the rules of their validator), `samples.parquet`, `tracklets.parquet` (every relation), `links.parquet`,
`verification.parquet` and a manifest. `--captions captions.jsonl` supplies each relation's
caption and the character spans of its objects; without it only `tracklets.parquet` is
filled. Exit code 1 when a captioned record fails validation or an annotation is missing (the
problems are in the manifest). Field mapping and open points: [docs/CONCOR_VIDEO.md](CONCOR_VIDEO.md).

```bash
vidstg-masks export-concor --worklist outputs/val/worklist.json --campaign-root outputs/val --captions captions.jsonl
```

## render

```
vidstg-masks render [roots] --vid VID --campaign-root DIR [--out FILE] [--alpha X] [--no-labels] [--side-by-side] [--crf N]
```

A QA video of one clip from its records: every written mask tinted in its object's colour
(`--alpha`, default 0.45), each object's VidOR box (thick at a human keyframe, thin at a
tracker box), a label `tid:category` at each box with `(no mask)` where the object has a box
but no mask (`--no-labels` turns the labels off), and a banner strip under the frame with the
video id, the relations and the colour of each object. `--side-by-side` writes the untouched frame on the
left and the painted one on the right, each captioned. H.264, `--crf` 20 by default (27 is
about half the size). Default output `<campaign>/overlays/<vid>.mp4`. CPU only; the frames
are never part of a release. Exit code 1 when the clip has no records.

```bash
vidstg-masks render --vid 7639717122 --campaign-root outputs/val --side-by-side --crf 27
```

## transcode

```
vidstg-masks transcode [roots] [--vids ...] [--vids-file FILE] [--worklist FILE [--campaign-root DIR] [--all-black]]
```

Re-encodes videos to H.264 into `$VIDOR_TRANSCODED_ROOT/<folder>/<vid>.mp4` without dropping
or duplicating frames (`-vsync 0`) and without changing the frame size (an odd width or height
is written with 4:4:4 chroma), then checks the result against the annotation: frame count,
size, not black. `resolve_video` prefers the transcoded file from then on. Which videos: the
ones named by `--vids` / `--vids-file`; with `--worklist --campaign-root`, every unit the
campaign refused as `decode_black` or `frame_size_mismatch`, whose refusal records (`runs/`,
`records/`, `errors/`) are then moved to `<campaign>/superseded/<kind>/` and whose worklist
unit is pointed at the new file, so the next `process` runs the clip; with `--worklist
--all-black`, every unit whose video decodes black (probed now). Requires
`VIDOR_TRANSCODED_ROOT`. Exit code 1 when a source file is missing or a re-encode fails the
frame-count, size or black check (that file is not used).

```bash
vidstg-masks transcode --worklist outputs/val/worklist.json --campaign-root outputs/val
```
