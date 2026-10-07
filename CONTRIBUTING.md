# Contributing

Focused pull requests are welcome. The map of the modules, the path of a clip through them and
where a change belongs is in [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md).

## Before opening a pull request

- `pytest -q` passes (CPU only, a few seconds; 124 tests on synthetic annotations and a tiny
  generated video). Add a test when changing dataset parsing (`datasets.py`), anchor selection
  (`anchors.py`, `anchor_quality.py`), the merge (`merge.py`), checkpoint recovery (`worker.py`:
  claims, errors, interruption, retries), the record contract (`records.py`,
  `schema/mask_record.schema.json`) or an export (`export.py`, `concor.py`).
- A new flag is wired in four places: `cli.py` (the parser and `cmd_process` / `cmd_process_one`),
  `worker.py` (`run_settings`, so it lands in every `runs/<vid>.json`, and the subprocess
  command), `scripts/run_local.sh`, `scripts/submit_slurm.sh` and `slurm/process_array.slurm`
  (an environment variable each), and the docs (`README.md` Settings, `docs/CLI.md`,
  `docs/SLURM.md`). `tests/test_worker.py` checks that the subprocess command carries the flag.
- A new refusal reason is added to `records.REASON_CODES`, the schema, and the table in
  `docs/OUTPUT_FORMAT.md`.
- Numbers in the docs come from a committed script or a recorded run; say which.

## What never goes into the repository

Dataset media, annotations, model weights, tokens, private paths, campaign outputs
(`outputs/`, any `CAMPAIGN_ROOT`), rendered videos or Slurm logs. `examples/*.env` hold
placeholders only. `.gitignore` already lists the usual suspects.

## The regression check

The masks are pinned byte-for-byte to the eager, no-FlashAttention SAM path, the pinned sam3
commit (`96914d2`) and the checkpoint hash (`0567debe...`). Any change to `sam_session.py`,
`sam3_compat.py`, the sam3 commit or the checkpoint needs the regression check against a known
clip: regenerate it and compare its `records/<vid>.jsonl` RLEs with the previous run before
merging. The same holds for `merge.py`: a change to the merge must either reproduce the previous
records on a `--direction both` clip or say, with counts, what it changed.
