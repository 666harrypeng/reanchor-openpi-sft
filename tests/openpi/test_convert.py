"""Writing the LeRobot v2.1 companion dataset with openpi's pinned lerobot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reanchor_sft import contract
from reanchor_sft.convert import SIDECAR, write_lerobot_v21
from tests.test_convert_hdf5_reader import H, W, write_fake_hdf5


def test_write_lerobot_v21_round_trips_frames_states_actions_and_task(
    tmp_path: Path, lerobot_home: Path
):
    root = lerobot_home  # lerobot captured this root at import (conftest)
    hdf5 = write_fake_hdf5(tmp_path / "src.hdf5", lengths=(6, 6))
    out = write_lerobot_v21(hdf5, repo_id="local/fake_v21", root=root, source_sha256="ab" * 32)
    assert out == root / "local" / "fake_v21" and not list(root.glob("local/.fake_v21.building"))

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/fake_v21", root=out)
    assert len(ds) == 12 and ds.fps == 25 and ds.num_episodes == 2
    item = ds[7]  # episode 1, frame 1
    np.testing.assert_allclose(item["state"].numpy(), [10, 11, 12, 1, 0, 0, 0, 1])
    np.testing.assert_allclose(item["actions"].numpy(), 2 + np.arange(8) / 10, rtol=1e-6)
    assert tuple(item["image"].shape) == (3, H, W) and tuple(item["wrist_image"].shape) == (3, H, W)
    assert float(item["wrist_image"].max()) > 0.5  # frames carry pixels (151/255), not zeros
    assert item["task"] == contract.TASK_PROMPT
    assert ds.meta.features["state"]["names"] == list(contract.STATE_NAMES)
    assert ds.meta.info["codebase_version"] == "v2.1"

    side = json.loads((out / SIDECAR).read_text())
    assert (side["episodes"], side["frames"], side["prompt"]) == (2, 12, contract.TASK_PROMPT)
    assert side["source_sha256"] == "ab" * 32 and side["lerobot_revision"].startswith("0cf8648")


def test_threaded_image_writer_matches_the_synchronous_dataset(tmp_path: Path, lerobot_home: Path):
    """LeRobot's asynchronous image writer (the natural parallelism: PNG encoding is the cost)
    produces frame-for-frame the dataset of the synchronous writer."""
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    hdf5 = write_fake_hdf5(tmp_path / "src.hdf5", lengths=(6, 5, 4, 6, 5))
    sync = write_lerobot_v21(
        hdf5, repo_id="local/w_sync", root=lerobot_home, source_sha256="0" * 64, writer_threads=0
    )
    thr = write_lerobot_v21(
        hdf5, repo_id="local/w_thr", root=lerobot_home, source_sha256="0" * 64, writer_threads=4
    )
    a = LeRobotDataset("local/w_sync", root=sync)
    b = LeRobotDataset("local/w_thr", root=thr)
    assert len(a) == len(b) == 26 and b.num_episodes == 5
    assert a.meta.info["total_frames"] == b.meta.info["total_frames"] == 26
    assert [e["length"] for e in b.meta.episodes.values()] == [6, 5, 4, 6, 5]
    for i in range(26):
        ia, ib = a[i], b[i]
        for key in ("state", "actions", "image", "wrist_image", "timestamp"):
            assert torch.equal(ia[key], ib[key]), (i, key)
        for key in ("index", "episode_index", "frame_index", "task_index"):
            assert int(ia[key]) == int(ib[key]), (i, key)
    assert json.loads((thr / SIDECAR).read_text())["writer_threads"] == 4
    assert not list(thr.glob("images"))  # the writer's PNG staging is gone after embedding
