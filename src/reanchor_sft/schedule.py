"""Launch request and schedule scaling.

The profile is written for one topology (global batch 256). Any other global batch keeps the
sample exposure: every step count of the recipe is scaled by `ceil(s × 256 / batch)`, so 4 × H100
at global 128 run 6000 updates with milestones 2000 / 4000 / 6000, and one 96 GB card at 32 runs
24 000. There is no gradient accumulation and no silent batch change: the launcher refuses a
batch the device count does not divide.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from reanchor_sft.profile import SFTProfile

Mode = Literal["smoke", "formal"]
_RUN_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
SMOKE_STEPS = 100
SMOKE_RESUME_STEPS = 120


@dataclass(frozen=True)
class SFTLaunchRequest:
    run_name: str
    mode: Mode
    resume: bool
    device_count: int
    batch_size: int

    def __post_init__(self) -> None:
        if not _RUN_NAME.match(self.run_name):
            raise ValueError(f"run_name must match {_RUN_NAME.pattern!r}: {self.run_name!r}")
        if self.mode not in ("smoke", "formal"):
            raise ValueError(f"mode must be 'smoke' or 'formal', got {self.mode!r}")
        if self.device_count < 1:
            raise ValueError("device_count must be ≥ 1")


@dataclass(frozen=True)
class SFTSchedule:
    num_train_steps: int
    rolling_save_interval: int
    milestone_interval: int
    checkpoint_steps: tuple[int, ...]
    batch_size: int
    warmup_steps: int
    decay_steps: int  # the cosine clock: the formal length, also under a smoke run
    per_device_batch: int


def resolve_schedule(profile: SFTProfile, request: SFTLaunchRequest) -> SFTSchedule:
    batch = request.batch_size
    if batch < request.device_count or batch % request.device_count:
        raise ValueError(
            f"global batch {batch} must be divisible by the device count {request.device_count}"
        )

    def scaled(steps: int) -> int:
        return math.ceil(steps * profile.batch_size / batch)

    milestone = scaled(profile.keep_period)
    decay = 3 * milestone
    warmup = scaled(profile.warmup_steps)
    per_device = batch // request.device_count
    if request.mode == "smoke":
        if request.resume:
            return SFTSchedule(
                SMOKE_RESUME_STEPS,
                20,
                20,
                (SMOKE_STEPS, SMOKE_RESUME_STEPS),
                batch,
                warmup,
                decay,
                per_device,
            )
        return SFTSchedule(
            SMOKE_STEPS, SMOKE_STEPS, SMOKE_STEPS, (SMOKE_STEPS,), batch, warmup, decay, per_device
        )
    return SFTSchedule(
        num_train_steps=decay,
        rolling_save_interval=scaled(profile.save_interval),
        milestone_interval=milestone,
        checkpoint_steps=(milestone, 2 * milestone, 3 * milestone),
        batch_size=batch,
        warmup_steps=warmup,
        decay_steps=decay,
        per_device_batch=per_device,
    )
