# Output format

```
$CAMPAIGN_ROOT/
  worklist.json          units (vid, split, relations, segment, paths), counts, roots
  records/<vid>.jsonl    one record per (tid, fid) with a VidOR box inside the tid span
  runs/<vid>.json        timing, VRAM, offload flag, mask / refusal counts, the run settings
                         (anchor_policy, max_anchors, hq_fallback, max_gap, gap_fill,
                         keep_span_edges, contained_negatives)
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
| prompt_payload | object | `anchor_fids`, `ref_anchor_fid`, `anchor_policy` (`human`, `human_gap`, `hq`); `hq` (gate results) under hq; `gap` (coverage-rule fills) under human_gap; `contained_negatives` (negative clicks, co-prompt frames) present unless `--no-contained-negatives`. Frames added by a co-prompt are included in `anchor_fids` |
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
