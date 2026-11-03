"""Client for ReflexBench's HTTP policy-server protocol.

The protocol (third_party/reflexbench/scripts/evaluation/POLICY_SERVER.md) has two endpoints:
`GET /info` returns the policy's metadata and `POST /predict` returns one action chunk per
environment with shape `(num_envs, action_horizon, action_dim)`. Two request types: `state` (the
flat state vector; privileged replay / scripted policies) and `vla` (JPEG q90 base64 camera
images, proprioception fields and the task description; learned policies). The field values sent
here mirror what ReflexBench's `eval.py` sends, so a server cannot tell the two harnesses apart.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import httpx
import numpy as np
from PIL import Image

# Mirrors eval.py `_CONTROL_TO_ACTION_FMT` and its default `state_format` / `orientation_rep`.
ACTION_FORMAT = {"joint_pos": "abs_joint", "ik_abs": "abs_eef_pose", "ik_rel": "rel_eef_pose"}
STATE_FORMAT = {"joint_pos": "joint", "ik_abs": "eef_pose", "ik_rel": "eef_pose"}
ORIENTATION_REP = "quat"
JPEG_QUALITY = 90  # eval.py encodes camera frames as JPEG at this quality before base64
PROPRIO_KEYS = ("gripper_state", "eef_pos", "eef_orient")


def encode_jpeg_b64(rgb: np.ndarray) -> str:
    """One `(H, W, 3)` uint8 RGB frame → base64 JPEG string, as eval.py does."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"frame must be (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}")
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass(frozen=True)
class PolicyInfo:
    action_dim: int
    action_horizon: int
    control_mode: str
    model_name: str


class PolicyClient:
    """Thin synchronous client; one instance per policy server."""

    def __init__(
        self, base_url: str, timeout_s: float = 60.0, client: httpx.Client | None = None
    ) -> None:
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)

    @classmethod
    def from_app(cls, app) -> PolicyClient:
        """Test-only constructor that talks to an ASGI app in-process (Starlette's TestClient)."""
        from starlette.testclient import TestClient

        return cls("http://testserver", client=TestClient(app))

    def info(self) -> PolicyInfo:
        r = self._client.get("/info")
        r.raise_for_status()
        d = r.json()
        return PolicyInfo(
            action_dim=int(d["action_dim"]),
            action_horizon=int(d["action_horizon"]),
            control_mode=str(d.get("control_mode", "")),
            model_name=str(d.get("model_name", "")),
        )

    def predict_state(
        self, state: np.ndarray, step_ids: list[int], control_mode: str
    ) -> np.ndarray:
        """Request one chunk per environment from a flat state observation `(N, S)`."""
        state = np.asarray(state, dtype=np.float32)
        if state.ndim != 2 or state.shape[0] != len(step_ids):
            raise ValueError(f"state must be (N, S) with N == len(step_ids); got {state.shape}")
        payload = {
            "type": "state",
            "num_envs": int(state.shape[0]),
            "state": state.tolist(),
            "control_mode": control_mode,
            "action_format": ACTION_FORMAT[control_mode],
            "state_format": STATE_FORMAT[control_mode],
            "orientation_rep": ORIENTATION_REP,
            "step_ids": [int(s) for s in step_ids],
        }
        r = self._client.post("/predict", json=payload)
        r.raise_for_status()
        actions = np.asarray(r.json()["actions"], dtype=np.float32)
        if actions.ndim != 3 or actions.shape[0] != state.shape[0]:
            raise ValueError(
                f"server returned actions of shape {actions.shape}, expected (N, H, D)"
            )
        return actions

    def predict_vla(
        self,
        images: dict[str, np.ndarray],
        proprio: dict[str, np.ndarray],
        prompt: str,
        step_ids: list[int],
        control_mode: str,
    ) -> np.ndarray:
        """Request one chunk per environment from camera frames `(N, H, W, 3)` per camera,
        proprioception arrays `(N, k)` (`gripper_state`, `eef_pos`, `eef_orient`) and the task."""
        n = len(step_ids)
        for name, frames in images.items():
            frames = np.asarray(frames)
            if frames.ndim != 4 or frames.shape[0] != n:
                raise ValueError(
                    f"images[{name!r}] must be (num_envs, H, W, 3); got {frames.shape}"
                )
        if set(proprio) != set(PROPRIO_KEYS):
            raise ValueError(f"proprio must have exactly {PROPRIO_KEYS}, got {sorted(proprio)}")
        for key, arr in proprio.items():
            arr = np.asarray(arr)
            if arr.ndim != 2 or arr.shape[0] != n:
                raise ValueError(f"proprio[{key!r}] must be (num_envs, k); got {arr.shape}")
        payload = {
            "type": "vla",
            "num_envs": n,
            "control_mode": control_mode,
            "action_format": ACTION_FORMAT[control_mode],
            "state_format": STATE_FORMAT[control_mode],
            "orientation_rep": ORIENTATION_REP,
            "proprioception": {
                k: np.asarray(proprio[k], dtype=np.float64).tolist() for k in PROPRIO_KEYS
            },
            "images": {
                name: [encode_jpeg_b64(f) for f in np.asarray(frames)]
                for name, frames in images.items()
            },
            "task_description": prompt,
            "step_ids": [int(s) for s in step_ids],
        }
        r = self._client.post("/predict", json=payload)
        r.raise_for_status()
        actions = np.asarray(r.json()["actions"], dtype=np.float32)
        if actions.ndim != 3 or actions.shape[0] != n:
            raise ValueError(
                f"server returned actions of shape {actions.shape}, expected (N, H, D)"
            )
        return actions

    def close(self) -> None:
        self._client.close()
