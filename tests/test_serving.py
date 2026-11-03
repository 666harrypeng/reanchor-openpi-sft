"""The π0.5 policy server app: ReflexBench's `vla` request in, `(N, 50, 8)` chunks out. The
policy is injected (a callable on the openpi observation dict), so this runs without openpi."""

from __future__ import annotations

import numpy as np
import pytest

from reanchor_sft.client import PolicyClient
from reanchor_sft import contract
from reanchor_sft.serving import build_pi05_app


class RecordingPolicy:
    def __init__(self):
        self.calls = []

    def __call__(self, obs: dict) -> np.ndarray:
        self.calls.append(obs)
        # a recognisable chunk: row i = i, last column = the state's gripper bit
        chunk = np.tile(np.arange(50, dtype=np.float32)[:, None], (1, 8))
        chunk[:, 7] = obs["observation/state"][7]
        return chunk


def _client(policy):
    return PolicyClient.from_app(build_pi05_app(policy, model_name="pi05-test"))


def test_info_matches_contract():
    info = _client(RecordingPolicy()).info()
    assert (info.action_dim, info.action_horizon, info.control_mode) == (8, 50, "ik_abs")
    assert info.model_name == "pi05-test"


def test_predict_decodes_images_and_orders_state():
    policy = RecordingPolicy()
    client = _client(policy)
    fixed = np.full((2, 48, 64, 3), 200, np.uint8)
    wrist = np.full((2, 48, 64, 3), 30, np.uint8)
    proprio = {
        "eef_pos": np.array([[0.4, 0.1, 0.3], [0.5, -0.1, 0.2]]),
        "eef_orient": np.array([[0.0, 1.0, 0.0, 0.0], [0.7, 0.0, 0.7, 0.0]]),
        "gripper_state": np.array([[1.0], [0.0]]),
    }
    out = client.predict_vla(
        {"fixed_cam": fixed, "wrist_cam": wrist}, proprio, contract.TASK_PROMPT, [3, 4], "ik_abs"
    )
    assert out.shape == (2, 50, 8)
    assert out[0, 10, 0] == 10 and out[0, 0, 7] == 1.0 and out[1, 0, 7] == 0.0
    assert len(policy.calls) == 2
    obs = policy.calls[1]
    # state in the dataset's order: eef_pos, eef_orient (wxyz), gripper_state
    np.testing.assert_allclose(obs["observation/state"], [0.5, -0.1, 0.2, 0.7, 0.0, 0.7, 0.0, 0.0])
    assert obs["observation/state"].dtype == np.float32
    assert (
        obs["observation/image"].shape == (48, 64, 3) and obs["observation/image"].dtype == np.uint8
    )
    assert abs(int(obs["observation/image"][0, 0, 0]) - 200) <= 3  # JPEG q90 round trip
    assert abs(int(obs["observation/wrist_image"][0, 0, 0]) - 30) <= 3
    assert obs["prompt"] == contract.TASK_PROMPT


def test_state_requests_and_wrong_cameras_are_rejected():
    client = _client(RecordingPolicy())
    with pytest.raises(Exception, match="400|vla"):
        client.predict_state(np.zeros((1, 27), np.float32), [0], "ik_abs")
    img = np.zeros((1, 8, 8, 3), np.uint8)
    proprio = {
        "eef_pos": np.zeros((1, 3)),
        "eef_orient": np.zeros((1, 4)),
        "gripper_state": np.ones((1, 1)),
    }
    with pytest.raises(Exception, match="400|camera"):
        client.predict_vla({"fixed_cam": img}, proprio, contract.TASK_PROMPT, [0], "ik_abs")


def test_reset_is_accepted():
    from starlette.testclient import TestClient

    r = TestClient(build_pi05_app(RecordingPolicy(), model_name="x")).post("/reset", json={})
    assert r.status_code == 200
