# ConCor Video export

`vidstg-masks export-concor` writes the mask tracks of every relation in the record contract
of [ConCor-Video-Data-Processing](https://github.com/suryathecreator/ConCor-Video-Data-Processing)
(schema `concor-video-tracklet-bcc-v2`), the pipeline this output feeds. The contract was
read from that repository on 2026-10-02 (`schema/tracklet_record.schema.json`,
`src/concor_video/tracklet_schema.py`, `exporter.py`, `docs/OUTPUT_FORMAT.md`).

## What a ConCor Video record is

One record is one text ↔ instance correspondence sample for one video: the text, the frame
ids, one **tracklet** per instance with a mask on every frame (uncompressed COCO RLE, `null`
where the instance has no mask), **groups** (tracklet → text spans, each with a role
`main_referent` or `context_entity` and an identity string), **span_links** (the
deterministic inverse, span → tracklets) and a **disposition** (`complete_bcc`,
`incomplete_context`, `missing_main_referent`, `negative_unsegmentable`). Their validator
fails a record whose spans do not slice the text exactly, whose tracklets are not all
linked, whose mask lists are not aligned to the frame ids, or whose RLE runs do not sum to
H × W. Export tables: `samples.parquet`, `tracklets.parquet` (one row per tracklet, masks
aligned to the frame ids as JSON), `links.parquet`, `verification.parquet` (self-contained
rows for their browser verifier), `run_ledger.csv`, `manifest.json`.

## Our mapping

| ConCor field | ours |
|---|---|
| `sample_id` | `vidstg:<split>:<vid>:<subject_tid>-<predicate>-<object_tid>`, one per relation |
| `dataset`, `cohort` | `vidstg`, `relation` — **not in their enums** (`ref_youtube_vos`, `revos`); needs their schema extended |
| `split` | the VidSTG split |
| `video_id`, `expression_id` | `<vid>`, `<subject_tid>-<predicate>-<object_tid>` |
| `text` | the relation's BCC-complete caption, from `--captions` (never the VidSTG sentence) |
| `frame_ids` | every frame of the relation segment, `%06d` in native frame index space |
| `frame_files` | `<vid>/%06d.jpg`, the convention for frames extracted with `-vsync 0`; no frames are written |
| `tracklets[]` | `vidor-<tid>` for the subject and the object; `source` `sam3.1_main_referent` (their enum has no value for SAM masks prompted from ground-truth boxes; the origin is in `source_annotation_id` = `vidor:<vid>:<tid>` and in `vidstg_provenance`); `confidence` = mean `mask_confidence` over masked frames, `max_confidence` the max; `present_frames`; `masks` aligned to `frame_ids`, `null` outside the object's boxed span and on refusals |
| `groups` | subject → `main_referent`, object → `context_entity`, identity = VidOR category, spans from `--captions` |
| `span_links` | rebuilt from `groups` |
| `disposition` | `complete_bcc` when both objects have a mask and a span; `incomplete_context` when the object has no mask (its tracklet is dropped and listed in `vidstg_refused_tracklets`); `missing_main_referent` when the subject has none |
| `negative` | always false (VidSTG has no nonexistent-object samples) |
| extra keys | `vidstg_provenance` per tracklet (model, checkpoint hash, prompt mode, anchor policy, anchors, refusal counts), `extraction.relation`, `sam_prompt_audit`, `pipeline`, `vidstg_refused_tracklets`; their schema allows additional keys |

A relation without a caption cannot be a valid record (the schema needs a text and a
main-referent group), so it appears in `tracklets.parquet` only, with an empty `text` and
`disposition`; the caption stage fills them. `manifest.json` counts both kinds.

## Captions file

`--captions captions.jsonl`, one row per relation:

```json
{"vid": "2406339050", "subject_tid": 0, "predicate": "above", "object_tid": 1,
 "text": "A baby in pink sits above a blue baby walker.",
 "spans": {"0": [[0, 6]], "1": [[26, 44]]}}
```

Spans are half-open character offsets into `text`, one list per VidOR tid, and must slice
the text exactly. A caption may name an entity only if it has a tracklet (BCC); the export
drops nothing silently: a span for an object without masks leaves the record
`incomplete_context`.

## Open points for the ConCor Video side

1. Add `vidstg` to the `dataset` enum (and accept the cohort `relation`).
2. A tracklet `source` value for SAM masks prompted from ground-truth boxes, or agreement
   that `sam3.1_main_referent` plus `source_annotation_id` is enough.
3. Whether the relation object should be `context_entity` or a second `main_referent`.
4. Frames: we list every frame of the relation segment; their samples use the dataset's
   sampled frames. If a sampled subset is wanted, it is a filter on `frame_ids`.

## Output

```
<campaign>/export/concor/
  records/<sample_id>.json      captioned relations only, each validated
  samples.parquet               their SAMPLE_SCHEMA
  tracklets.parquet             their TRACKLET_SCHEMA, every relation (captioned or not)
  links.parquet                 their LINK_SCHEMA
  verification.parquet          their VERIFICATION_SCHEMA
  manifest.json                 counts, dispositions, problems, open points
```
