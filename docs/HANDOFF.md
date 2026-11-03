# Handoff: π0.5 SFT on the conveyor static demonstrations

This document is everything needed to run the formal training and hand the checkpoints back.

## 1. What is being trained

- **Policy**: π0.5 (openpi), full-parameter fine-tuning from `pi05_base`, pi0.5 discrete state
  input, action horizon 50, flow-matching action expert; only the 8 real action dimensions carry
  loss (`active_action_dim: 8`).
- **Data**: 200 successful teleop-free demonstrations (32 094 frames, 25 Hz) of "pick up the cube
  from the conveyor and place it into the bin". Per frame: `image` (fixed camera, 480 × 640),
  `wrist_image`, `state` = `x y z qw qx qy qz gripper` (end-effector pose + gripper state, 1 open /
  0 closed), `actions` = the same 8 names as an absolute end-effector pose + gripper command.
  Dataset: `yypeng666/reanchor-conveyor-static-ikabs-v2-lerobot21` (LeRobot v2.1; gated — request
  access on the Hub, it is granted manually).
- **Recipe** (`configs/sft/pi05_conveyor_static.yaml`, do not edit): global batch 256, 3000
  updates (≈ 23.9 epochs), warmup 150, cosine 5e-5 → 5e-6, AdamW clip 1.0, EMA 0.999, bf16, seed
  42. Milestones at 1000 / 2000 / 3000 updates are kept; a rolling checkpoint every 250.

## 2. Hardware and environment

- Reference topology: **8 × H100 80 GB, global batch 256** (32 per card, ≈ 70 GB peak each).
  4 × H100: use `--batch-size 128`; the schedule rescales itself to 6000 updates with milestones
  2000 / 4000 / 6000 (same sample exposure). Any CUDA ≥ 12.2 driver works (the wheels bring CUDA 12).
- One machine, one process: the launcher detects the GPUs (`jax.device_count()`); the global
  batch must be divisible by that count; there is no gradient accumulation and the launcher never
  changes the batch on its own. If a batch does not fit, choose a smaller global batch — the
  schedule follows.
- Verified so far: the whole chain (preflight → 100 updates → resume → receipts → W&B) on one
  RTX PRO 6000 at global batch 32; the 8 × H100 formal run is the step this document hands over.
- Environment: `bash tools/openpi/sync.sh` creates `tools/openpi/.venv` from openpi's frozen lock
  (JAX 0.5.3, flax 0.10.2, lerobot at openpi's pin) plus a few extras. Or build
  `docker/sft.Dockerfile`. Disk: ≈ 15 GB dataset, 12 GB base checkpoint, ≈ 25 GB per milestone
  with optimizer state (`runs/`).
- Credentials, environment variables only: `HF_TOKEN` (dataset access, publishing),
  `WANDB_API_KEY` (W&B project `reanchor-sft`; the run name is `--run-name`). Caches stay inside
  the repository (`.cache/`).

## 3. Steps

```
git clone --recurse-submodules https://github.com/666harrypeng/reanchor-openpi-sft.git
cd reanchor-openpi-sft && bash tools/openpi/sync.sh
PY=tools/openpi/.venv/bin/python
JAX_PLATFORMS=cpu $PY -m pytest tests -q                       # ≈ 1 min, CPU only

# dataset: download the gated LeRobot v2.1 dataset into data/lerobot/<repo id>
$PY -c "from huggingface_hub import snapshot_download as d; d('yypeng666/reanchor-conveyor-static-ikabs-v2-lerobot21', repo_type='dataset', local_dir='data/lerobot/yypeng666/reanchor-conveyor-static-ikabs-v2-lerobot21')"

# 1. preflight (data, transforms, norm stats; no training)
$PY scripts/sft_train.py --run-name formal-h100x8 --work-dir runs/formal --mode formal \
    --dataset-root data/lerobot --batch-size 256 --check-data-only
# 2. smoke: 100 updates in a separate work dir, then a resume to 120
$PY scripts/sft_train.py --run-name smoke --work-dir runs/smoke --mode smoke --dataset-root data/lerobot --batch-size 256
$PY scripts/sft_train.py --run-name smoke --work-dir runs/smoke --mode smoke --dataset-root data/lerobot --batch-size 256 --resume
# 3. formal (≈ 1.5 h on 8 × H100: the reference pipeline measured 1.6 s per update at 32 per card; W&B on)
$PY scripts/sft_train.py --run-name formal-h100x8 --work-dir runs/formal --mode formal --dataset-root data/lerobot --batch-size 256
# 4. publish the three milestones (public listing, manual gate)
$PY scripts/sft_publish.py --checkpoint-dir runs/formal/checkpoints/pi05_conveyor_static/formal-h100x8 \
    --steps 1000 2000 3000 --repo <your user>/reanchor-pi05-conveyor-static-v1
```

`--resume` continues an interrupted run from its last rolling checkpoint; it requires the same
`--run-name`, `--work-dir`, `--batch-size` and GPU count (the launch contract next to the
checkpoint directory is compared). A new attempt with different settings needs a new run name.

## 4. What comes back

1. The Hub repository id with the three milestones (`<step>/params`, `<step>/assets`,
   `_CHECKPOINT_METADATA`, licences, model card, the receipt).
2. `runs/formal/checkpoints/pi05_conveyor_static/formal-h100x8/verified_run-step3000.json` (the
   receipt: code revision, dataset and profile hashes, schedule, topology, elapsed time, device
   memory) and `runs/formal/preflight.json`.
3. The W&B run link (project `reanchor-sft` under your own entity, run `formal-h100x8`): loss,
   grad norm, param norm, learning rate.

## 5. Troubleshooting

- *`global batch ... must be divisible by the device count`*: pass a `--batch-size` the GPU count
  divides, or `--device-count` to use a subset (set `CUDA_VISIBLE_DEVICES` accordingly).
- *OOM*: lower the global batch (e.g. 8 × 24 = 192); the schedule rescales (4002 updates).
- *`Repository Not Found` for the dataset*: the dataset must sit under `--dataset-root/<repo id>`
  with its `meta/` and `data/` directories; check the `snapshot_download` target.
- *Resume refused*: the contract differs; see the key named in the error.
- *Blackwell consumer / workstation cards (RTX 50xx, RTX PRO 6000, compute capability 12.0)*: the
  pinned toolchain's Triton GEMM path fails in XLA's autotuner (`ptxas ... syntax error`). Set
  `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`; training then runs but several times slower
  than on H100. Data-centre Hopper (H100, H200) does not need this.
- *Checkpoint download*: `pi05_base` (12 GB) is fetched from Google Cloud Storage anonymously
  into `.cache/openpi` on the first run; set `OPENPI_DATA_HOME` to relocate the cache.
