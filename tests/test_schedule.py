"""Schedule scaling: a different global batch keeps the sample exposure of the reference recipe
(`ceil(s × 256 / batch)` for steps, warmup, milestones and the rolling save interval)."""

from __future__ import annotations

import pytest

from reanchor_sft.profile import load_profile
from reanchor_sft.schedule import SFTLaunchRequest, SFTSchedule, resolve_schedule

P = load_profile()


def req(device_count: int, batch: int, mode: str = "formal", resume: bool = False):
    return SFTLaunchRequest(
        run_name="formal-h100x8",
        mode=mode,
        resume=resume,
        device_count=device_count,
        batch_size=batch,
    )


@pytest.mark.parametrize(
    "devices,batch,steps,milestones,warmup,rolling",
    [
        (8, 256, 3000, (1000, 2000, 3000), 150, 250),
        (4, 128, 6000, (2000, 4000, 6000), 300, 500),
        (1, 32, 24000, (8000, 16000, 24000), 1200, 2000),
        (4, 192, 4002, (1334, 2668, 4002), 200, 334),
    ],
)
def test_formal_schedule_preserves_sample_exposure(
    devices, batch, steps, milestones, warmup, rolling
):
    s = resolve_schedule(P, req(devices, batch))
    assert isinstance(s, SFTSchedule)
    assert s.num_train_steps == steps and s.decay_steps == steps
    assert s.checkpoint_steps == milestones and s.milestone_interval == milestones[0]
    assert s.warmup_steps == warmup and s.rolling_save_interval == rolling
    assert s.batch_size == batch and s.per_device_batch == batch // devices
    assert s.batch_size * s.num_train_steps >= P.training_examples


def test_smoke_and_resume_schedules():
    s = resolve_schedule(P, req(1, 32, mode="smoke"))
    assert (s.num_train_steps, s.rolling_save_interval, s.milestone_interval) == (100, 100, 100)
    assert s.checkpoint_steps == (100,)
    r = resolve_schedule(P, req(1, 32, mode="smoke", resume=True))
    assert (r.num_train_steps, r.rolling_save_interval, r.milestone_interval) == (120, 20, 20)
    assert r.checkpoint_steps == (100, 120)
    # the learning-rate clock is the formal one, so a smoke run exercises the real warmup
    assert s.warmup_steps == 1200 and s.decay_steps == 24000


def test_uneven_or_tiny_batch_is_refused():
    with pytest.raises(ValueError, match="divisible"):
        resolve_schedule(P, req(8, 100))
    with pytest.raises(ValueError, match="divisible"):
        resolve_schedule(P, req(8, 4))


def test_run_name_and_mode_are_validated():
    with pytest.raises(ValueError, match="run_name"):
        SFTLaunchRequest("Bad Name", "formal", False, 1, 32)
    with pytest.raises(ValueError, match="mode"):
        SFTLaunchRequest("ok", "fast", False, 1, 32)
