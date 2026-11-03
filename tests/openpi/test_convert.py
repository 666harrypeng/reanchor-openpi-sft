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
