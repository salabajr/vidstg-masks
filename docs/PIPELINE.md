# Pipeline

## Unit of work

One VidOR video. Its VidSTG records (6.6 on average) are joined on `vid`; the object set
is the union of `subject_tid` and `object_tid` over all their `used_relation`s; the clip span
is the interval hull of their `used_segment`s. Per object, the span is the clip span
intersected with the object's VidOR box presence. One SAM 3.1 Object Multiplex session
segments every object of the clip together, so the masks of one frame are disjoint.

## Worklist (CPU, `worker.build_worklist`)

`build-worklist` loads the requested VidSTG file(s), indexes the VidOR annotations (a
recursive search for `<digits>.json` under `VIDOR_ANN_ROOT`, any folder depth; a missing
annotation is reported with the pattern searched) and asserts the dataset counts before
writing anything: VidSTG records per requested split (train 36,202 / val 3,996 / test 4,610;
44,808 in all) and 7,835 VidOR annotation files. A mismatch raises and no worklist is written;
`--no-assert-counts` skips the check for a partial copy. `worklist.json` then carries one
unit per video (relations, segment hull, categories, `n_tids`, `n_frames`), the dataset
roots, each unit's absolute `vidor_ann` and `video_path` as resolved on the building node,
and `counts` (`units`, `by_split`, `video_missing`, `too_many_objects`, `prop_frame_objects`).

`process` and `process-one` read the roots and paths from the worklist by default, so a
worker needs no data variables in its environment. `--roots-from-env` replaces the frozen
roots with the environment variables / `--*-root` flags and rebases every unit's paths onto
them: for a worklist built on a login node whose mount points differ from the compute nodes,
or data that moved after the build. Without it a moved dataset fails every clip with
`FileNotFoundError`. `submit_slurm.sh` sets the flag with `ROOTS_FROM_ENV=1` and skips the
count assertion with `ASSERT_COUNTS=0`.

## Planning (CPU, `anchors.plan_clip`)

1. Boxes come from `trajectories[fid]` of the VidOR JSON; `generated` (0 human, 1 tracker)
   and `tracker` are propagated into every record, never recomputed.
2. Anchors are the human keyframes (`generated == 0`) inside the object span. The reference
   anchor is the keyframe whose box has the smallest maximum IoU with any other relation
   object's box at that frame (tie: earliest). This seeds identity where the object is most
   separable.
3. At most `--max-anchors` (16) anchors per object: the reference plus the rest thinned
   evenly (first and last human keyframe kept). Prompt-phase VRAM scales with this count.
4. An object with no human keyframe in its span is refused (`no_human_keyframe`) before any
   GPU work.
5. `--anchor-policy hq` filters the human keyframes through `anchor_quality.py`: boxes
   touching a frame edge, overlapping another object's box, moving fast between neighbouring
   keyframes, shrunk relative to the object's typical size, or blurred (Laplacian variance)
   are flagged; flag-free keyframes are kept, else the least-flagged (`--hq-fallback
   least-flagged`, recorded in provenance) or the object is refused (`no_hq_anchor`).
6. `--anchor-policy human_gap` (the default) applies the coverage rule of step 5 to the human
   plan without the gate (`anchor_quality.human_gap_plan`); it needs no video. Contained
   negatives (on by default, any policy; `--no-contained-negatives` turns them off) add
   negative clicks and co-prompts for objects that contain another relation object
   (`anchors.add_contained_negatives`). `--anchor-policy human --no-contained-negatives` is
   the plan of steps 1-4 alone, the default before human_gap (see "Anchor policies").

## Pre-checks (CPU, `worker.precheck_clip`)

- more than 16 relation objects: `too_many_objects` (the multiplex bucket size).
- no video file: `video_missing`.
- one sequential OpenCV decode: frame count must equal the annotation's `frame_count`
  (`frame_count_mismatch`), size must equal `width x height` (`frame_size_mismatch`), and
  eight evenly spaced frames must not all be black (`decode_black`).

The pre-checks run on the human plan; the hq gate (which reads the video) is applied only
after they pass. A clip refused here has never been prompted, so its refusal records carry
an empty `anchor_fids` list and the requested `anchor_policy`.

Frame indices are native decode order; the pipeline never resamples or indexes by time.

## SAM session (`sam_session.run_session`)

```
start_session(resource_path=<video>, offload_video_to_cpu=True)
for each object (ascending tid): add_prompt at its reference anchor
    points=[[x0,y0],[x1,y1]] (relative), point_labels=[2,3]   # box corners on the instance path
    purge_prompt_buffers(keep=this frame)
for each remaining anchor sorted by (frame, tid): add_prompt + purge
    # contained negatives (on by default): clicks ride in the same add_prompt,
    # points=corners + [[x,y], ...], point_labels=[2,3] + [0, ...]; at most 16 points
purge_prompt_buffers(keep=None)
propagate_in_video(forward, start=span start, max_frame_num_to_track=span length)
    # --direction backward: propagate_in_video(backward, start=span end, same count)
collect_sam2_scores()   # per-frame tracker logits -> sigmoid confidence
close_session()         # always, in a finally
```

`sam3_compat.py` carries the patches this needs against facebookresearch/sam3 at commit
96914d2: `start_session` kwargs filtering, `_build_sam2_output` returning refined masks on
frames without detector cache, `offload_state_to_cpu` plumbing for the inner tracker,
detector buffer purging between scattered prompt frames, and a device fix in the multiplex
merge. All are applied by `apply_pvs_patches` in `build_predictor`.

State offload (tracker outputs on the CPU) is switched on when the propagation span exceeds
1,500 frames or frames x objects exceeds 2,500; it costs 10-15% speed and avoids OOM on
24 GB. A clip that still OOMs is retried once with offload forced on.

### Backward pass and the merge (`--direction`)

SAM's memory is freshest just after an anchor, so a forward pass fails most often on the
frames just before the next anchor. `--direction backward` runs the same prompts from the
span end in its own session (a second propagation inside one session only re-runs objects
with new prompts, so it has to be a new session); SAM 3.1 then predicts the frames from the
span end minus one down to the span start, and the span-end frame is recorded as `not_tracked`.

`--direction both` (the default) runs the forward pass, keeps it as RLE, runs the backward
pass and merges them frame by frame (`merge.merge_passes`, the pixels rule; no box is used):

1. Each object's two candidates are read. A mask under `--speck-floor` pixels (20) is no mask.
   With two real masks, one under `--speck-ratio` (0.1) of the other is no mask. If one pass is
   left, it is the only candidate: a strong vote. If both are left and their overlap is at or
   above `--agree-iou` (0.3), the masks mostly match: the forward one is preferred, either is
   allowed (a weak vote). If both are left and they do not overlap, the frame is a dispute and
   is refused (`disputed_mask`, rule 7). Nothing real in either pass is `empty_mask` (both empty)
   or `speck_mask`.
2. The objects of a frame are decided together. Each takes its candidate; every pixel that two
   written masks share goes to the object whose pass was the only candidate, and the object that
   merely preferred forward loses those pixels (its own backward mask, from the winner's pass,
   also leaves them out). Two strong votes from different passes on the same pixels refuse both
   objects (`passes_conflict`). A trimmed mask that falls under the floor takes the object's
   backward mask when that touches nothing written, else it is refused (`handed_over_speck`).
   Inside one pass SAM gives a pixel to one object only, so after the handover no two written
   masks share a pixel.

Every record of a `both` run says `direction: bidirectional` and carries `prompt_payload.merge`:
the reading (`rule`), the `decision` (`forward`, `backward`, `refused`, `none`), the `source`
pass, the overlap of the two masks, each candidate's size, share inside the VidOR box (for the
reader; unused) and confidence, the `handover` when pixels moved, the `refused_reason`, and the
thresholds. `both` costs twice the GPU time of `forward`. On the 20 review clips of the
research run (69,729 object-frames): 217 refusals, 222 object-frames with nothing real in
either pass, 176 masks from the backward pass, 180 masks trimmed, no two masks sharing a pixel;
the earlier per-object rule with a box tie-break had left 282 overlapping pairs and taken
one-pixel masks (`reports/merge_rules_v2.md` there).

## Records (`worker.clip_records`)

For each object and each frame inside its span that carries a VidOR box: a mask record
(compressed COCO RLE, confidence) or a refusal (`empty_mask` when SAM returns no pixels,
`not_tracked` when the pass never reached the frame).
A video-level defect refuses every frame of every object with its reason. Every record
carries the full provenance set (`docs/OUTPUT_FORMAT.md`).

## Worker (`worker.process`)

Each shard walks the whole worklist from its own offset and takes what nobody has claimed:
`claims/<vid>.claim` is created with `O_EXCL`, heartbeats every 30 s and is considered dead
after 30 min. A finished clip is `records/<vid>.jsonl` (written to `.part` and renamed);
finished clips are skipped on every pass, so the same campaign can be resumed or re-submitted.
Each clip runs in a fresh subprocess (`process-one`) because SAM 3.1 does not release session
VRAM between clips. Under Slurm the checkpoint sha256 is computed once per task and passed to
each subprocess (`docs/SLURM.md`).

A unit is `done` (records exist), `failed` (`errors/<vid>.json` holds an error that gets no
further attempt) or `pending` (anything else, including a clip another shard holds).
`errors/<vid>.json` records the last failure (attempt, error_type, message, traceback,
`cuda_oom`, and `interrupted` when the subprocess was killed by a signal or the shard was
draining):

- CUDA out of memory: one retry with the tracker state offloaded to the CPU
  (`--force-offload`).
- Subprocess killed by a signal (Slurm walltime `SIGTERM`, preemption, `scancel`), or running
  while the shard was draining: recorded with `"interrupted": true`, stays `pending`, retried
  on the next pass (after the requeue), 5 attempts in total. Previously an interrupted clip was
  recorded as terminally failed, which lost one clip per task at every walltime boundary.
- Any other error: terminal after one attempt; shows as `failed` in the ledger with its
  `error_type`. Delete the error file to retry.

`SIGUSR1`/`SIGTERM` to the shard: stop claiming, finish the active clip if it can, exit 99
(Slurm requeue). `vidstg-masks status` prints the ledger counts of a running campaign without
writing anything.

## Export (`export.export_campaign`)

`records/*.jsonl` become `masks.parquet` and `refusals.parquet` (streamed, zstd), plus
`ledger.csv` (every worklist unit: done, failed or pending; written atomically) and
`manifest.json`. The integrity check recomputes each clip's boxed in-span (tid, fid) set from
the VidOR annotation and requires exactly one record for each, decodes every RLE, and
validates every record against the schema. Problems are listed in the manifest and make
`export` exit 1.

## Anchor policies

After the gate, a coverage rule (`anchor_quality._fill_gaps`) guarantees that no stretch of
an object's span — between consecutive anchors, before the first, after the last — is longer
than `Thresholds.max_gap` frames (default 60): the least-flagged human keyframe nearest the
middle of the largest gap is put back until every gap fits; with `gap_fill = "any"` a tracker
box is used where no human keyframe lies in the gap. Each fill is recorded in
`prompt_payload.hq.gap_fills`. Without it, an object whose keyframes mostly fail the gate
loses its mask for hundreds of frames (SAM's memory judges it absent until the next anchor);
on the 20 VidSTG-val clips that was 2,265 frames, 1,353 of them beyond the first or last
kept anchor.


`--anchor-policy human_gap` (default): each object is prompted with its VidOR box at up to 16
human-annotated keyframes (`generated == 0`) inside its span; the reference keyframe is the
one whose box overlaps other relation objects least. The tracker boxes (about 97% of VidOR
boxes) are never shown to SAM. The coverage rule then applies with no quality gate — every
human keyframe counts as clean, so a fill is simply the human keyframe nearest the middle of
the largest gap (or a tracker box with `--gap-fill any`). It needs no video, keeps the human
policy's reference anchor and its `prompt_mode` (`pvs_box_multianchor`), and records the
fills in `prompt_payload.gap` (`max_gap`, `gap_fill`, `gap_fills`, `n_gap_fills`,
`baseline_fids`). In the research pipeline it was the ablation that asked whether the gate
adds anything over consistent spacing alone.

`--anchor-policy human`: the same keyframes and reference anchor without the coverage rule
(the `baseline_fids` of a human_gap payload are exactly this plan's anchors); the default
before human_gap. Same `prompt_mode`; the payload has no `gap` entry.

`--anchor-policy hq`: the keyframes filtered by a quality gate
(`src/vidstg_masks/anchor_quality.py`: frame-edge contact, overlap with other objects, motion,
relative size, Laplacian blur), then the coverage rule; an object with no clean keyframe keeps
its least-flagged ones (default) or is refused with `no_hq_anchor` under `--hq-fallback
refuse`. The gate's decisions (kept, dropped and why, thresholds) are recorded in each
record's `prompt_payload.hq`, and `prompt_mode` becomes `pvs_box_hqanchor`. The gate
thresholds were set on 16 VidSTG-val clips and have not been validated at scale.

`--max-gap` (default 60; 0 disables the rule) and `--gap-fill` (`human`, or `any` to allow a
tracker box where no human keyframe lies in the gap) set the coverage rule under human_gap
and hq; `--keep-span-edges` (hq only) always anchors the object's first and last human
keyframe in its span, whatever the gate says (SAM tracks forward: the frames before the first
anchor and after the last are the first to be lost).

Contained negatives (on by default, any policy; `--no-contained-negatives` turns them off):
overlap-aware prompts from the VidOR boxes alone. At every anchor frame of an object A, each
other relation object B whose box lies inside A's box (at least 90% of B's area) and is small
relative to it (at most 25% of A's area) adds a negative click for A at B's box centre; and at
every anchor frame of such a B, A is prompted with its own box plus that click as well, so the
multiplex layer resolves the shared pixels with both objects' prompts present. Without them
the larger object annexes the smaller one between the smaller one's prompts. The clicks travel
in the same `add_prompt` as the box corners (a second prompt on the same object and frame
would replace the first), and 2 corners + clicks may not exceed the tracker's 16-point cap.
Recorded per object in `prompt_payload.contained_negatives` (`neg_clicks` per frame as
`[x, y, tid_B]`, `co_prompt_fids`, `co_prompt_tracker_fids`, `n_neg_clicks`); the co-prompt
frames are added to `anchor_fids`. Under `--no-contained-negatives` the prompts are the box
corners alone and the payload has no `contained_negatives` entry.

`--anchor-policy human --no-contained-negatives` together reproduce the default before
human_gap: the same prompts as before. Every `runs/<vid>.json` records `anchor_policy`,
`max_anchors`, `hq_fallback`, `max_gap`, `gap_fill`, `keep_span_edges` and
`contained_negatives`; `vidstg-masks status` lists the distinct settings found in a campaign.

## Glossary

- **unit, clip** — one VidOR video with all its VidSTG records joined on `vid`: one entry
  of `worklist.json`, one `records/<vid>.jsonl`, one GPU subprocess.
- **relation** — a VidSTG `used_relation`: `(subject_tid, predicate, object_tid)` over a
  frame interval. Both tids are segmented.
- **tid** — VidOR trajectory id: one object of one video. Its records form one mask track.
- **fid** — frame index in native decode order. Never a timestamp; never rescaled.
- **span** — per tid: the clip's segment hull (interval hull of its records' `used_segment`s)
  intersected with the frames where the tid has a VidOR box. Every boxed frame in the span
  gets exactly one record.
- **prop_span** — the hull of all tid spans of a clip: the interval SAM propagates over,
  forward only.
- **anchor** — a frame at which an object is prompted with its VidOR box (as two corner
  points); at most `--max-anchors` (16) per object before the gap rule puts keyframes back.
- **reference anchor** — the anchor prompted first for an object: the human keyframe whose
  box has the smallest maximum IoU with any other relation object's box (tie: earliest fid).
- **human keyframe** — a VidOR box with `generated == 0`, drawn by an annotator (about 3% of
  boxes). The only boxes ever shown to SAM; the rest are tracker boxes.
- **hq gate** — `--anchor-policy hq`: the filter over human keyframes (`edge`, `overlap`,
  `motion`, `small`, `blur` flags) deciding which of them may be anchors.
- **gap rule, gap fill** — the coverage rule after anchor selection: no stretch of an object's
  span longer than `--max-gap` frames without an anchor; a fill is the keyframe put back to
  satisfy it (recorded in `prompt_payload.hq.gap_fills` or `prompt_payload.gap.gap_fills`).
- **human_gap** — `--anchor-policy human_gap` (the default): the human policy plus the gap
  rule, no gate. Same prompt type as `human`. `--anchor-policy human` is the keyframes alone.
- **contained negatives** — on by default, off with `--no-contained-negatives`: a negative
  click (label 0) for a container object at the centre of a small object boxed inside it, at
  the container's own anchors and, co-prompted with its box, at the contained object's anchors.
- **campaign** — one `CAMPAIGN_ROOT`: a `worklist.json` and everything written from it
  (`records/`, `runs/`, `errors/`, `claims/`, `export/`, `slurm_logs/`). Resumed by path.
- **shard** — one `vidstg-masks process` run: one GPU, one Slurm array task. Shards share the
  worklist and claim clips with lease files.
- **refusal, reason code** — a record with `rle: null` and a `reason_code`, written where no
  mask is emitted (table in `docs/OUTPUT_FORMAT.md`). Preferred to a doubtful mask.
- **Silver** — model-generated, not human-verified. Everything this pipeline writes is
  Silver; it is never mixed into a file with human-verified (Gold) masks.
