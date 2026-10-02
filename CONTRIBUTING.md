# Contributing

Focused pull requests are welcome. Add a fixture or a unit test when changing dataset parsing
(`datasets.py`), anchor selection (`anchors.py`), the quality gate (`anchor_quality.py`), checkpoint
recovery (`worker.py`: claims, errors, interruption, retries) or the export schema (`export.py`,
`schema/mask_record.schema.json`). Run `pytest -q` (CPU-only, about 20 s) before opening a pull request.

Never commit dataset media, annotations, model weights, tokens, private paths, campaign outputs
(`outputs/`, any `CAMPAIGN_ROOT`) or Slurm logs. `examples/*.env` hold placeholders only.

The masks are pinned byte-for-byte to the eager, no-FlashAttention SAM path, the pinned sam3 commit
(`96914d2`) and the checkpoint hash (`0567debe...`). Any change to `sam_session.py`, `sam3_compat.py`,
the sam3 commit or the checkpoint needs the regression check against a known clip: regenerate it and
compare its `records/<vid>.jsonl` RLEs with the previous run before merging.
