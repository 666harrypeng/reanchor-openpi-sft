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
    tools/openpi/thp_off.py SCRIPT   # run SCRIPT with transparent huge pages off (see below)

Tests that need openpi live in `tests/openpi/` and run with
environment skips that directory.

Checkpoint saves that never finish (zero bytes written, one thread at 100 % kernel time) on a
multi-socket host with automatic NUMA balancing and THP are the kernel migrating the ~40 GB save
buffers between NUMA nodes under compaction (seen on a 2-socket RTX PRO 6000 node, 2026-11-03).
Run the trainer through `tools/openpi/thp_off.py`, which disables THP for the process and its
workers; the 42 GB train-state save then takes under a minute.
