# Datasets

## VidSTG

Source: github.com/Guaranteer/VidSTG-Dataset (`annotations/{train,val,test}_annotations.json`).
44,808 records over 6,770 videos (train 36,202 / val 3,996 / test 4,610 records; each video
belongs to exactly one file). A record has `vid`, `used_segment {begin_fid, end_fid}`,
`used_relation {subject_tid, predicate, object_tid, begin_fid, end_fid}`, `temporal_gt`,
`subject/objects [{tid, category}]`, `captions`, `questions`. Records carry no boxes.

The unique (vid, tid) pairs covering subject and object of every `used_relation` number
26,016. The VidSTG sentences are not used by this pipeline.

```
$VIDSTG_ROOT/annotations/train_annotations.json
$VIDSTG_ROOT/annotations/val_annotations.json
$VIDSTG_ROOT/annotations/test_annotations.json
```

(the three files directly under `$VIDSTG_ROOT` are found as well)

`--split` selects which of these files the worklist is drawn from (`all` = every video).
The split name is written into each record as `split`.

## VidOR

Source: huggingface.co/datasets/shangxd/vidor (`training-annotation.zip`,
`validation-annotation.zip`, and the video archives). 7,835 per-video annotation JSONs, of
which 6,770 are VidSTG videos. A JSON has `video_id`, `video_path` (`<folder>/<vid>.mp4`),
`frame_count`, `fps`, `width`, `height`, `subject/objects`, `trajectories` (one list per frame
of `{tid, bbox {xmin, ymin, xmax, ymax}, generated, tracker}`), `relation_instances`.

`generated` is 0 for a human-annotated keyframe box and 1 for a box the annotation tool's
tracker (`tracker`: kcf, mosse, linear, ...) interpolated; about 97% of boxes are tracker
boxes. This pipeline prompts SAM only with human boxes and copies both fields into every
record.

```
$VIDOR_ANN_ROOT/training/<folder>/<vid>.json      (the release layout; the index searches
$VIDOR_ANN_ROOT/validation/<folder>/<vid>.json     every <digits>.json under the root, any depth)
$VIDOR_VIDEO_ROOT/<folder>/<vid>.mp4
```

Frame indices in both annotation sets are native-fps decode indices. `doctor --vid <vid>`
and `process` decode each video and require the frame count to equal `frame_count`.

## Videos that decode black

About 2.8% of VidOR videos are VP6F-encoded (some 500x375) and decode as black frames under
OpenCV. `process` refuses them with `decode_black`. `vidstg-masks transcode` re-encodes them
without dropping or duplicating frames (`ffmpeg -vsync 0`, H.264) and without changing the
frame size (an odd width or height is written with 4:4:4 chroma) into
`$VIDOR_TRANSCODED_ROOT/<folder>/<vid>.mp4`, which `resolve_video` prefers over the raw file.
With `--worklist --campaign-root` it picks the refused clips itself, moves their refusal
records to `<campaign>/superseded/` and points their worklist units at the new files, so the
next `process` runs them ([docs/CLI.md](CLI.md#transcode)).
