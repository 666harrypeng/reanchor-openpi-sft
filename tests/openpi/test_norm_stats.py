"""Norm stats through the patched openpi loader: every frame counted once, partial batch kept."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reanchor_sft.profile import load_profile
from reanchor_sft.schedule import SFTLaunchRequest


def test_norm_stats_count_every_frame_once(patched_openpi, tiny_dataset, tmp_path, lerobot_home):
    from reanchor_sft.training import build_train_config, compute_norm_stats

    cfg = build_train_config(
        load_profile(),
        SFTLaunchRequest("stats", "smoke", False, 1, 8),
        dataset_repo_id=tiny_dataset,
        assets_root=tmp_path / "assets",
        checkpoint_root=tmp_path / "ckpt",
        wandb_enabled=False,
    )
    with pytest.raises(ValueError, match="11"):
        compute_norm_stats(cfg, expected_frames=11)
    out = compute_norm_stats(
        cfg, expected_frames=12
    )  # 12 frames, batch 8: the partial batch counts
    assert out == cfg.assets_dirs / tiny_dataset / "norm_stats.json" and out.exists()
    stats = json.loads(out.read_text())["norm_stats"]
    # the fixture's states: episode k has eef_pos 10k + (0,1,2), quat (1,0,0,0), gripper t % 2
    states = np.concatenate(
        [
            np.column_stack(
                [
                    np.full((6, 3), 10 * k) + np.arange(3),
                    np.tile([1, 0, 0, 0], (6, 1)),
                    (np.arange(6) % 2)[:, None],
                ]
            )
            for k in range(2)
        ]
    )
    np.testing.assert_allclose(stats["state"]["mean"], states.mean(0), atol=1e-5)
    actions = np.concatenate([np.full((6, 8), k + 1) + np.arange(8) / 10 for k in range(2)])
    np.testing.assert_allclose(stats["actions"]["mean"], actions.mean(0), atol=1e-5)
    # the config's data factory now loads them
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert data_config.norm_stats is not None and set(data_config.norm_stats) == {
        "state",
        "actions",
    }
    assert Path(lerobot_home / tiny_dataset).exists()
