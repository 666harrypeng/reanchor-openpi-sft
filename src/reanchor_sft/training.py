"""openpi `TrainConfig` for the conveyor recipe, norm stats and the in-process trainer entry.

Everything here imports openpi lazily and expects the patched copy on `sys.path`
(`reanchor_sft.source.temporary_patched_openpi_copy`). Pure data parallel: one JAX process,
`fsdp_devices=1`; the global batch must divide the device count (`reanchor_sft.schedule`).
"""

from __future__ import annotations

import importlib.util
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from reanchor_sft import contract
from reanchor_sft.profile import SFTProfile
from reanchor_sft.schedule import SFTLaunchRequest, resolve_schedule

CONFIG_NAME = "pi05_conveyor_static"
PROJECT_NAME = "reanchor-sft"  # W&B project; the run name is the launch request's run_name
SEED = 42


def build_train_config(
    profile: SFTProfile,
    request: SFTLaunchRequest,
    *,
    dataset_repo_id: str,
    assets_root: Path,
    checkpoint_root: Path,
    wandb_enabled: bool,
) -> Any:
    import openpi.models.pi0_config as pi0_config
    import openpi.training.optimizer as optimizer
    import openpi.training.weight_loaders as weight_loaders
    from openpi.training.config import TrainConfig

    from reanchor_sft.data import ConveyorDataConfig

    s = resolve_schedule(profile, request)
    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=contract.ACTION_HORIZON,
        discrete_state_input=True,
        active_action_dim=contract.ACTION_DIM,
    )
    return TrainConfig(
        name=CONFIG_NAME,
        project_name=PROJECT_NAME,
        exp_name=request.run_name,
        model=model,
        data=ConveyorDataConfig(repo_id=dataset_repo_id),
        weight_loader=weight_loaders.CheckpointWeightLoader(profile.base_checkpoint),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=s.warmup_steps,
            peak_lr=profile.peak_learning_rate,
            decay_steps=s.decay_steps,
            decay_lr=profile.decay_learning_rate,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=profile.ema_decay,
        assets_base_dir=str(Path(assets_root).resolve()),
        checkpoint_base_dir=str(Path(checkpoint_root).resolve()),
        seed=SEED,
        batch_size=s.batch_size,
        num_workers=profile.num_workers,
        num_train_steps=s.num_train_steps,
        log_interval=profile.log_interval,
        save_interval=s.rolling_save_interval,
        keep_period=s.milestone_interval,
        overwrite=False,
        resume=request.resume,
        wandb_enabled=wandb_enabled,
        fsdp_devices=profile.fsdp_devices,
        policy_metadata={
            "profile_id": profile.profile_id,
            "dataset_repo_id": dataset_repo_id,
            "fps": contract.FPS,
            "state_names": list(contract.STATE_NAMES),
            "action_names": list(contract.STATE_NAMES),
            "action_horizon": contract.ACTION_HORIZON,
            "active_action_dim": contract.ACTION_DIM,
            "cameras": dict(contract.CAMERAS),
            "task_prompt": contract.TASK_PROMPT,
            "mode": request.mode,
            "training_parallelism": "replicated_data_parallel",
            "fsdp_devices": profile.fsdp_devices,
            "training_device_count": request.device_count,
            "per_device_batch": s.per_device_batch,
            "reference_training_examples": profile.training_examples,
            "training_examples": s.batch_size * s.num_train_steps,
            "checkpoint_steps": list(s.checkpoint_steps),
            "openpi_revision": profile.openpi_revision,
        },
    )


def register_config(config: Any) -> None:
    """Make the config visible to openpi's own entry points (`get_config`, norm-stats script)."""
    from openpi.training.config import _CONFIGS_DICT

    _CONFIGS_DICT[config.name] = config


def _openpi_copy_root() -> Path:
    import openpi

    return Path(openpi.__file__).resolve().parents[2]


def _load_script(name: str):
    path = _openpi_copy_root() / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"_reanchor_openpi_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


NORMALIZED_ABS_LIMIT = 3.0  # quantile-normalized data lives in about [-1, 1]; positions reach ~1.5


def identity_quaternion_norm(stats: Any) -> Any:
    """Return `stats` with the quaternion dims (3:7) normalized by the identity.

    π0.5 maps [q01, q99] → [-1, 1]. The expert's orientation is one constant top-down quaternion on
    every row but the planner's first (the hold at the current pose), so qw and qz have
    q01 == q99 == 0 and that row would be scaled by ~1e5 (the 2026-11-03 smoke logged a loss of
    6e6). Unit-quaternion components already lie in [-1, 1]: (q01, q99) = (-1, 1) and
    (mean, std) = (0, 1) leave them unchanged under either normalization scheme.
    """
    q = contract.QUATERNION_DIMS
    fields = (stats.mean, stats.std, stats.q01, stats.q99)
    mean, std, q01, q99 = (np.array(a, copy=True) for a in fields)
    mean[q], std[q], q01[q], q99[q] = 0.0, 1.0, -1.0, 1.0
    return type(stats)(mean=mean, std=std, q01=q01, q99=q99)


def normalized_extremes(
    stats: Any,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    names: tuple[str, ...],
    limit: float = NORMALIZED_ABS_LIMIT,
) -> dict:
    """Per-dimension (min, max) of the raw extremes after quantile normalization.

    Raises when a dimension leaves ±`limit`: a near-constant dimension with outliers, which the
    quantile scheme would blow up into the training targets.
    """
    q01, q99 = np.asarray(stats.q01, dtype=np.float64), np.asarray(stats.q99, dtype=np.float64)
    scale = q99 - q01 + 1e-6
    n_lo = (np.asarray(lo, dtype=np.float64) - q01) / scale * 2.0 - 1.0
    n_hi = (np.asarray(hi, dtype=np.float64) - q01) / scale * 2.0 - 1.0
    out: dict = {name: (float(a), float(b)) for name, a, b in zip(names, n_lo, n_hi, strict=True)}
    bad = {name: out[name] for name in names if max(abs(out[name][0]), abs(out[name][1])) > limit}
    if bad:
        raise ValueError(
            f"normalized extremes beyond ±{limit}: {bad} — a near-constant dimension with "
            "outliers under quantile normalization; its training targets would be scaled by the "
            "reciprocal of the quantile range"
        )
    out["max_abs"] = float(np.max(np.abs(np.concatenate([n_lo, n_hi]))))
    return out


def compute_norm_stats(config: Any, *, expected_frames: int) -> Path:
    """State / action statistics over every frame once (patch 0001 keeps the partial batch).

    Quaternion dims use the identity (`identity_quaternion_norm`); the raw extremes are checked
    after normalization (`normalized_extremes`) so a blown-up target can never reach training.
    """
    import openpi.shared.normalize as normalize

    data_config = config.data.create(config.assets_dirs, config.model)
    script = _load_script("compute_norm_stats.py")
    loader, _ = script.create_torch_dataloader(
        data_config,
        config.model.action_horizon,
        min(config.batch_size, expected_frames),
        config.model,
        0,
    )
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    lo: dict[str, np.ndarray] = {}
    hi: dict[str, np.ndarray] = {}
    frames = 0
    for batch in loader:
        state = np.asarray(batch["state"])
        actions = np.asarray(batch["actions"])
        mask = np.asarray(batch["action_loss_mask"])
        if state.ndim != 2 or actions.ndim != 3 or mask.shape != actions.shape:
            raise ValueError("unexpected norm-stat batch shapes")
        if not mask[:, 0].all():
            raise ValueError("every source must start with a real action")
        frames += state.shape[0]
        for key, rows in (("state", state), ("actions", actions[:, 0])):  # each frame counted once
            stats[key].update(rows)
            lo[key] = rows.min(0) if key not in lo else np.minimum(lo[key], rows.min(0))
            hi[key] = rows.max(0) if key not in hi else np.maximum(hi[key], rows.max(0))
    if frames != expected_frames:
        raise ValueError(f"norm stats covered {frames} frames, expected {expected_frames}")
    result = {k: identity_quaternion_norm(v.get_statistics()) for k, v in stats.items()}
    for key, value in result.items():
        extremes = normalized_extremes(value, lo[key], hi[key], names=contract.STATE_NAMES)
        logging.info("normalized %s extremes: %s", key, extremes)
    out_dir = Path(config.assets_dirs) / data_config.repo_id
    normalize.save(out_dir, result)
    return out_dir / "norm_stats.json"


def run_training(config: Any, *, device_count: int) -> None:
    """Call the patched `scripts/train.py` in this process after the topology checks."""
    import jax

    if jax.process_count() != 1:
        raise RuntimeError("one JAX process per run (replicated data parallel)")
    if jax.device_count() != device_count:
        raise RuntimeError(
            f"jax sees {jax.device_count()} devices, the request says {device_count}"
        )
    if config.batch_size % device_count or config.fsdp_devices != 1:
        raise RuntimeError("global batch must divide the device count; fsdp_devices must be 1")
    if math.gcd(config.batch_size, device_count) != device_count:
        raise RuntimeError("global batch must divide the device count")
    register_config(config)
    _load_script("train.py").main(config)
