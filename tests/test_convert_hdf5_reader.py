"""Reading the static HDF5 into episodes for the LeRobot v2.1 conversion (no lerobot needed)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from reanchor_sft import contract
from reanchor_sft.convert import Episode, iter_episodes, read_header

H, W = 8, 12


def write_fake_hdf5(
    path: Path, lengths=(5, 4), prompt=contract.TASK_PROMPT, fps=25.0, success=None
) -> Path:
    success = tuple(True for _ in lengths) if success is None else success
    with h5py.File(path, "w") as f:
        attrs = {"prompt": prompt, "fps": fps, "control_mode": "ik_abs", "action_dim": 8}
        attrs |= {"action_format": "abs_eef_pose", "state_format": "eef_pose"}
        attrs |= {"orientation_rep": "quat", "total_demos": len(lengths)}
        f.attrs.update(attrs)
        g = f.create_group("data")
        for k, (n, ok) in enumerate(zip(lengths, success, strict=True)):
            d = g.create_group(f"demo_{k}")
            d.attrs["num_samples"] = n
            d.attrs["success"] = ok
            d["actions"] = np.full((n, 8), k + 1, np.float32) + np.arange(8, dtype=np.float32) / 10
            d["obs/eef_pos"] = np.full((n, 3), 10 * k, np.float32) + np.arange(3, dtype=np.float32)
            d["obs/eef_orient"] = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
            d["obs/gripper_state"] = np.arange(n, dtype=np.float32)[:, None] % 2
            d["obs/fixed_cam_rgb"] = np.full((n, H, W, 3), 50 + k, np.uint8)
            d["obs/wrist_cam_rgb"] = np.full((n, H, W, 3), 150 + k, np.uint8)
    return path


def test_iter_episodes_assembles_state_in_contract_order(tmp_path: Path):
    eps = list(iter_episodes(write_fake_hdf5(tmp_path / "a.hdf5")))
    assert [e.name for e in eps] == ["demo_0", "demo_1"]
    e = eps[1]
    assert isinstance(e, Episode)
    assert e.state.shape == (4, 8) and e.state.dtype == np.float32
    # eef position, quaternion wxyz, gripper state
    np.testing.assert_array_equal(e.state[1], [10, 11, 12, 1, 0, 0, 0, 1])
    np.testing.assert_allclose(e.actions[0], 2 + np.arange(8) / 10, rtol=1e-6)
    assert e.fixed_cam.shape == (4, H, W, 3) and e.wrist_cam[0, 0, 0, 0] == 151


def test_header_and_demo_order_follow_the_index_not_the_string(tmp_path: Path):
    p = write_fake_hdf5(tmp_path / "b.hdf5", lengths=tuple([3] * 11))
    assert [e.name for e in iter_episodes(p)][-2:] == ["demo_9", "demo_10"]
    h = read_header(p)
    assert h.prompt == contract.TASK_PROMPT and h.fps == 25.0 and h.episodes == 11


def test_reader_refuses_a_different_prompt_or_fps_and_failed_demos(tmp_path: Path):
    with pytest.raises(ValueError, match="prompt"):
        list(iter_episodes(write_fake_hdf5(tmp_path / "c.hdf5", prompt="other")))
    with pytest.raises(ValueError, match="fps"):
        list(iter_episodes(write_fake_hdf5(tmp_path / "d.hdf5", fps=50.0)))
    with pytest.raises(ValueError, match="success"):
        list(iter_episodes(write_fake_hdf5(tmp_path / "e.hdf5", success=(True, False))))


def test_episode_limit(tmp_path: Path):
    p = write_fake_hdf5(tmp_path / "f.hdf5", lengths=(3, 3, 3))
    assert len(list(iter_episodes(p, limit=2))) == 2
