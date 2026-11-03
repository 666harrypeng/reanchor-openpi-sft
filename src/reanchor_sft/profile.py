"""The frozen SFT profile: the reference recipe every run is scaled from.

`configs/sft/pi05_conveyor_static.yaml` is the only place the recipe lives. A run with a
different global batch keeps the sample exposure (`reanchor_sft.schedule`); nothing else moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from reanchor_sft.source import OPENPI_REVISION, REPO_ROOT

PROFILE_PATH = REPO_ROOT / "configs" / "sft" / "pi05_conveyor_static.yaml"
WARMUP_FRACTION = 0.05


@dataclass(frozen=True)
class SFTProfile:
    profile_id: str
    openpi_revision: str
    patch_sha256: tuple[str, str, str]
    base_checkpoint: str
    dataset_repo_id: str
    source_frames: int
    batch_size: int
    num_train_steps: int
    warmup_steps: int
    peak_learning_rate: float
    decay_learning_rate: float
    save_interval: int
    keep_period: int
    log_interval: int
    num_workers: int
    fsdp_devices: int
    ema_decay: float

    def __post_init__(self) -> None:
        if self.openpi_revision != OPENPI_REVISION:
            raise ValueError(
                "profile openpi_revision must match reanchor_sft.source.OPENPI_REVISION"
            )
        if self.num_train_steps != 3 * self.keep_period:
            raise ValueError("num_train_steps must be 3 × keep_period (three milestones)")
        if not 0 < self.save_interval < self.keep_period:
            raise ValueError("save_interval must be positive and below keep_period")
        if self.warmup_steps != math.ceil(WARMUP_FRACTION * self.num_train_steps):
            raise ValueError(f"warmup_steps must be ceil({WARMUP_FRACTION} × num_train_steps)")
        if self.fsdp_devices != 1:
            raise ValueError("fsdp_devices must be 1: pure data parallel (decision 0014)")
        if self.batch_size <= 0 or self.batch_size % 8:
            raise ValueError("batch_size must be a positive multiple of 8")
        if self.decay_learning_rate >= self.peak_learning_rate:
            raise ValueError("decay_learning_rate must be below peak_learning_rate")

    @property
    def milestones(self) -> tuple[int, int, int]:
        k = self.keep_period
        return (k, 2 * k, 3 * k)

    @property
    def training_examples(self) -> int:
        return self.batch_size * self.num_train_steps

    @property
    def equivalent_epochs(self) -> float:
        return self.training_examples / self.source_frames


def load_profile(path: Path = PROFILE_PATH) -> SFTProfile:
    raw = yaml.safe_load(Path(path).read_text())
    raw["patch_sha256"] = tuple(raw["patch_sha256"])
    return SFTProfile(**raw)
