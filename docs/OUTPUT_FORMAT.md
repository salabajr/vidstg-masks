# Output format

```
$CAMPAIGN_ROOT/
  worklist.json          units (vid, split, relations, segment, paths), counts, roots
  records/<vid>.jsonl    one record per (tid, fid) with a VidOR box inside the tid span
  runs/<vid>.json        timing, VRAM, offload flag, mask / refusal counts, the run settings
                         (anchor_policy, max_anchors, hq_fallback, max_gap, gap_fill,
                         keep_span_edges, contained_negatives, direction, agree_iou,
                         speck_floor, speck_ratio, dispute_score, dispute_rule, dispute_winner) and,
                         for a `--direction both` run,
                         `merge`: how many object-frames each reading, decision and refusal reason got
  errors/<vid>.json      last failure (attempt, error_type, message, traceback, cuda_oom, interrupted)
  claims/<vid>.claim     transient lease while a worker holds the video
  overlays/<vid>.mp4     QA renders (vidstg-masks render); never part of a release
  export/
    masks.parquet        one row per mask
    refusals.parquet     one row per refusal
    ledger.csv           one row per worklist unit: done / failed / pending
    manifest.json        counts, reason codes, code commits, checkpoint hashes, integrity
```

## Record (records/*.jsonl, schema/mask_record.schema.json)

| field | type | meaning |
|---|---|---|
| vid | string | video id |
| tid | int | VidOR trajectory id |
| fid | int | frame index, native decode order |
| rle | object or null | `{size: [H, W], counts: <compressed COCO RLE>}`; null on refusal |
| model | string | `sam3.1-object-multiplex` |
| checkpoint_hash | string | sha256 of `sam3.1_multiplex.pt`, asserted at load |
| sam_version | string | `3.1` |
| prompt_mode | string | `pvs_box_multianchor` (human and human_gap policies) or `pvs_box_hqanchor` (hq) |
| prompt_payload | object | `anchor_fids`, `ref_anchor_fid`, `anchor_policy` (`human`, `human_gap`, `hq`); `hq` (gate results) under hq; `gap` (coverage-rule fills) under human_gap; `contained_negatives` (negative clicks, co-prompt frames) present unless `--no-contained-negatives`. Frames added by a co-prompt are included in `anchor_fids`. `direction`: `forward`, `backward`, or `bidirectional` for a `--direction both` run, whose records also carry `merge`: the reading of the two candidates (`rule`: `both_empty`, `forward_only`, `backward_only`, `backward_speck`, `forward_speck`, `both_speck`, `forward_speck_only`, `backward_speck_only`, `agree`, `dispute`), the `decision` (`forward`, `backward`, `refused`, `none`), the `source` pass, the overlap of the two masks, each candidate's size, share of pixels inside the VidOR box (for the reader; the merge uses no box) and confidence, `handover` (pixels given to a neighbour: `removed_px`, `to`, `area_before`, `area_after`, `fallback`) when pixels moved, `refused_reason` when refused, the thresholds `agree_iou`, `speck_floor_px`, `speck_ratio`, the tie-break settings `dispute_rule` (`refuse`, `forward_score`, `higher_score`), `dispute_score` (null under `refuse`) and `dispute_winner` (`weak`, `strong`), and on a disputed frame of a run with a tie-break rule, `tiebreak` (`rule`, `threshold`, `forward_score`, `backward_score`, `pick`, `taken`, `winner`) |
| mask_confidence | float or null | sigmoid of the per-frame tracker score; 1.0 on prompted frames; null on refusal |
| box_generated | 0 or 1 | VidOR flag at this frame: 0 human keyframe, 1 tracker box |
| box_tracker | string | VidOR tracker name at this frame (`none` for human boxes) |
| split | string | VidSTG file the video belongs to: train, val or test |
| code_commit | string | git commit of this package (`-dirty` if modified) |
| created_at | string | UTC ISO-8601 |
| reason_code | string | refusals only, see below |

Reason codes:

| code | scope | meaning |
|---|---|---|
| empty_mask | frame | SAM returned no pixels at a frame that has a VidOR box |
| no_human_keyframe | object | no human box inside the object span; identity cannot be seeded |
| no_hq_anchor | object | hq policy with `--hq-fallback refuse`: no keyframe passed the gate |
| frame_count_mismatch | clip | decoded frame count differs from the annotation |
| frame_size_mismatch | clip | decoded size differs from the annotation |
| decode_black | clip | video decodes black under OpenCV; run `transcode` |
| too_many_objects | clip | more than 16 relation objects |
| video_missing | clip | no file at the annotation's `video_path` |
| not_tracked | frame | the pass never predicted this frame: the span-end frame of a backward pass |
| disputed_mask | frame | `--direction both`: the forward and backward masks are both real and overlap below `--agree-iou` |
| passes_conflict | frame | `--direction both`: two objects' only candidates come from different passes and share pixels |
| handed_over_speck | frame | `--direction both`: a mask trimmed of a neighbour's pixels fell under `--speck-floor` and its backward mask touched a written one |
| speck_mask | frame | `--direction both`: the only mask either pass had was under `--speck-floor` pixels |

## Parquet tables

`masks.parquet`: `vid, tid, fid, rle_size_h, rle_size_w, rle_counts, mask_confidence` plus
the provenance columns `model, checkpoint_hash, sam_version, prompt_mode, prompt_payload
(JSON string), box_generated, box_tracker, split, code_commit, created_at`.
`refusals.parquet`: `vid, tid, fid, reason_code` plus the same provenance columns.
Decode a mask with pycocotools: `mask.decode({"size": [h, w], "counts": counts.encode()})`.

## Integrity (manifest.json["integrity"])

For every completed clip: each (tid, fid) with a VidOR box inside the tid span has exactly
one record; no record lies outside that set; every RLE decodes to the annotation's H x W;
every record validates against the schema. `ok` is false and `vidstg-masks export` exits 1
when any problem is found; the first 200 problems are listed.

## ConCor Video export (`vidstg-masks export-concor`)

`<campaign>/export/concor/`: one record per relation in the contract of
ConCor-Video-Data-Processing (schema `concor-video-tracklet-bcc-v2`) plus its
`samples.parquet`, `tracklets.parquet`, `links.parquet` and `verification.parquet` layouts.
The masks of a tracklet are uncompressed COCO RLE aligned to the relation's frame ids, `null`
where the object has no mask; a relation gets a full, validated record only when `--captions`
supplies its BCC-complete caption and spans, otherwise it is in `tracklets.parquet` alone.
Field mapping, the captions file and the open points: `docs/CONCOR_VIDEO.md`.
