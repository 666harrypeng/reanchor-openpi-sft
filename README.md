# reanchor-openpi-sft

π0.5 (openpi) full-parameter supervised fine-tuning on the **reanchor conveyor static**
demonstrations: a Franka picks a cube from a conveyor and places it in a bin, recorded at 25 Hz
with two 640 × 480 cameras, absolute end-effector pose actions. Start with
[docs/HANDOFF.md](docs/HANDOFF.md).

| item | value |
| --- | --- |
| dataset | `yypeng666/reanchor-conveyor-static-ikabs-v2-lerobot21` (LeRobot v2.1, gated: request access) |
| base checkpoint | `gs://openpi-assets/checkpoints/pi05_base/params` (downloaded automatically) |
| recipe | `configs/sft/pi05_conveyor_static.yaml`: global batch 256, 3000 updates, milestones 1000 / 2000 / 3000 |
| parallelism | replicated data parallel, one process, `batch % devices == 0`, ≈ 32 per 80 GB card |
| outputs | milestones with `params/`, `assets/norm_stats.json`, a run receipt; published with `scripts/sft_publish.py` |

```
bash tools/openpi/sync.sh                                   # one-time environment (CUDA 12 wheels)
PY=tools/openpi/.venv/bin/python
$PY scripts/sft_train.py --run-name smoke --work-dir runs/smoke --mode smoke --batch-size 32 --wandb-disabled
$PY scripts/sft_train.py --run-name formal-h100x8 --work-dir runs/formal --mode formal --batch-size 256
$PY scripts/sft_publish.py --checkpoint-dir runs/formal/checkpoints/pi05_conveyor_static/formal-h100x8 --steps 1000 2000 3000 --repo <user>/<repo>
```

## Licence

The code in this repository is released under the Apache License 2.0 (`LICENSE`). The π0.5 base
model and the fine-tuned checkpoints derived from it are subject to their own terms
(`licenses/pi05/LICENSE`, `NOTICE`, `GEMMA_TERMS.txt`), which are also attached to every published
checkpoint.
