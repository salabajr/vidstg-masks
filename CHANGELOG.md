# Changelog

Newest first. Versions are the `version` in `pyproject.toml`; between releases the entries are
grouped by branch.

## Unreleased (branch `backward-merge`)

**Backward pass and the merge.** `--direction {forward,backward,both}` (default `forward`).
`both` runs the same prompts a second time from the span end in its own SAM session and merges
the two passes frame by frame with the pixels rule: a mask under `--speck-floor` (20 px) is no
mask, a real mask under `--speck-ratio` (0.1) of the other pass's is no mask, two real masks
that overlap below `--agree-iou` (0.3) are a dispute, pixels two objects' masks share go to the
object whose pass was the only candidate, and two strong votes from different passes on the
same pixels refuse both. No box is read, no two masks on a frame share a pixel. New reason codes
`not_tracked`, `disputed_mask`, `speck_mask`, `passes_conflict`, `handed_over_speck`. Every record
of a `both` run carries `prompt_payload.merge`; every `runs/<vid>.json` the merge counts.

**Dispute tie-break.** `--dispute-rule {refuse,forward_score,higher_score}` (default `refuse`),
`--dispute-score T`, `--dispute-winner {weak,strong}` (default `weak`): on a disputed frame,
write the pass whose presence score (`mask_confidence`) clears `T`, as a weak or a strong vote
in the handover. The measured setting, `higher_score` at 0.907 with the strong winner, agreed
with the reviewer's frame labels on 14 of 16 firm cases on the 20 review clips
(`docs/PIPELINE.md`, Backward pass and the merge). Recorded as `merge.tiebreak`.

**Transcode.** `transcode` keeps the frame size (an odd width or height is written with 4:4:4
chroma; it used to pad by one pixel, and the pre-check then refused the repaired clip as
`frame_size_mismatch`). `transcode --worklist --campaign-root` selects the clips the campaign
refused as `decode_black` or `frame_size_mismatch`, verifies each re-encode against the
annotation, moves the refusal records to `<campaign>/superseded/` and points the worklist unit
at the new file, so the next `process` runs the clip. New `video.probe_size`.

**Records.** The validator no longer rejects the records of a `both` run for objects that were never
prompted (a clip-level refusal, or an object without a human keyframe): they say `bidirectional`
and carry no merge block. Before, `export` listed them as invalid and left them out of the tables.

**Render.** A label `tid:category` at each box, `(no mask)` where the object has a box but no
mask; `--side-by-side` (the untouched frame on the left, the painted one on the right);
`--crf`; `--no-labels`.

**Launchers.** `DIRECTION`, `AGREE_IOU`, `SPECK_FLOOR`, `SPECK_RATIO`, `DISPUTE_RULE`,
`DISPUTE_SCORE`, `DISPUTE_WINNER` in `run_local.sh`, `submit_slurm.sh` and `process_array.slurm`;
`SLURM_ACCOUNT` is passed only when set; `submissions.txt` records the direction and the
dispute settings.

**Docs.** `docs/SLURM.md` Sizing: host memory of a `both` run (about 14 MB per frame-object with
the tracker state offloaded), capping the tasks per node with `CPUS_PER_TASK` where `--mem` is
not enforced, `STAGE_CHECKPOINT` on a RAM-backed `/tmp`. New `docs/GETTING_STARTED.md`,
`docs/CLI.md`, `docs/CODE_STRUCTURE.md`, this changelog; the README rewritten around them.

Tests: 124.

## 0.1.0 (branch `main`)

The first release: worklist, claims, the shard worker with a fresh subprocess per clip,
pre-checks, the `human`, `human_gap` (default) and `hq` anchor policies, contained negatives,
one forward SAM 3.1 Object Multiplex pass, records with full provenance and refusal codes,
`export` with the integrity check, `export-concor`, `render`, `transcode`, `doctor`, `status`,
`plan`, the local and Slurm launchers. 103 tests.
