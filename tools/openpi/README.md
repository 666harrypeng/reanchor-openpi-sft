# openpi tool environment

π0.5 training and serving run in `tools/openpi/.venv`, built by `bash tools/openpi/sync.sh` from the
openpi submodule's frozen lock (`third_party/openpi/uv.lock`: JAX 0.5.3 with CUDA 12 wheels, flax

openpi itself is never edited: `reanchor_sft.source.temporary_patched_openpi_copy` applies
`patches/openpi/000{1,2,3}` to a `git archive` copy of the pinned commit for the duration of a run.

Entry points (all with `tools/openpi/.venv/bin/python`):

    scripts/sft_convert_dataset.py   # HDF5 -> LeRobot v2.1 (openpi's loader cannot read v3.0)
    scripts/sft_train.py             # norm stats, preflight, launch contract, training, receipt
    scripts/sft_publish.py           # inference milestones -> Hugging Face (public, gated manual)
    scripts/serve_pi05.py            # ReflexBench `vla` policy server for a checkpoint

Tests that need openpi live in `tests/openpi/` and run with
environment skips that directory.
