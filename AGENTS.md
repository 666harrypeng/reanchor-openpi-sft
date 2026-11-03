# reanchor-openpi-sft — agent notes

- `third_party/openpi` is a pinned, untouched submodule; changes to openpi are patches in
  `patches/openpi/`, applied to a temporary copy by `reanchor_sft.source`.
- The recipe lives in `configs/sft/pi05_conveyor_static.yaml` only. The global batch is a launch
  flag; every other number scales from it (`reanchor_sft.schedule`). No gradient accumulation.
- Tests: `JAX_PLATFORMS=cpu tools/openpi/.venv/bin/python -m pytest tests -q` must pass.
- Credentials (`HF_TOKEN`, `WANDB_API_KEY`) are read from the environment, never written to files.
- Publish with `scripts/sft_publish.py` (inference files only), never whole run directories.
