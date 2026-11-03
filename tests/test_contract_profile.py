"""The SFT contract constants (read from the dataset, never re-derived) and the frozen profile
that pins the reference recipe (global batch 256 × 3000 updates, milestones 1000/2000/3000)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from pathlib import Path

import pytest

from reanchor_sft import contract
from reanchor_sft.profile import PROFILE_PATH, SFTProfile, load_profile

REPO = Path(__file__).resolve().parents[1]


def test_contract_matches_the_dataset():
    assert contract.FPS == 25
    assert contract.STATE_NAMES == ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")
    assert contract.STATE_DIM == contract.ACTION_DIM == 8
    assert contract.ACTION_HORIZON == 50
    assert contract.TASK_PROMPT == "pick up the cube from the conveyor and place it into the bin"
    assert contract.CAMERAS == {"fixed_cam": "base_0_rgb", "wrist_cam": "left_wrist_0_rgb"}
    assert contract.IMAGE_HW == (480, 640)
    assert contract.SOURCE_FRAMES == 32094 and contract.SOURCE_EPISODES == 200
    assert contract.DATASET_REPO_V21 == "yypeng666/reanchor-conveyor-static-ikabs-v2-lerobot21"
    assert len(contract.SOURCE_HDF5_SHA256) == 64


def test_profile_loads_reference_values():
    p = load_profile(PROFILE_PATH)
    assert p.profile_id == "pi05_conveyor_static_v1"
    assert p.openpi_revision == "15a9616a00943ada6c20a0f158e3adb39df2ccac"
    assert p.base_checkpoint == "gs://openpi-assets/checkpoints/pi05_base/params"
    assert (p.batch_size, p.num_train_steps, p.warmup_steps) == (256, 3000, 150)
    assert (p.peak_learning_rate, p.decay_learning_rate) == (5e-5, 5e-6)
    assert (p.save_interval, p.keep_period, p.log_interval) == (250, 1000, 25)
    assert (p.num_workers, p.fsdp_devices, p.ema_decay) == (8, 1, 0.999)
    assert p.dataset_repo_id == contract.DATASET_REPO_V21
    assert p.source_frames == contract.SOURCE_FRAMES
    assert p.milestones == (1000, 2000, 3000)
    assert p.training_examples == 768_000
    assert p.equivalent_epochs == pytest.approx(768_000 / 32094)


def test_profile_rejects_steps_not_three_milestones():
    p = load_profile(PROFILE_PATH)
    with pytest.raises(ValueError, match="keep_period"):
        replace(p, num_train_steps=2999)


def test_profile_rejects_warmup_off_five_percent():
    p = load_profile(PROFILE_PATH)
    with pytest.raises(ValueError, match="warmup"):
        replace(p, warmup_steps=200)
    assert p.warmup_steps == math.ceil(0.05 * p.num_train_steps)


def test_profile_rejects_fsdp_and_save_interval_past_milestone():
    p = load_profile(PROFILE_PATH)
    with pytest.raises(ValueError, match="fsdp"):
        replace(p, fsdp_devices=2)
    with pytest.raises(ValueError, match="save_interval"):
        replace(p, save_interval=1000)


def test_patch_sha256_match_the_files():
    p = load_profile(PROFILE_PATH)
    names = (
        "0001-filter-incomplete-action-chunks.patch",
        "0002-save-completed-step-checkpoints.patch",
        "0003-mask-action-tails.patch",
    )
    digests = tuple(
        hashlib.sha256((REPO / "patches" / "openpi" / n).read_bytes()).hexdigest() for n in names
    )
    assert p.patch_sha256 == digests
    assert isinstance(p, SFTProfile)
