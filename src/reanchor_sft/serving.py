"""π0.5 policy server: ReflexBench's `vla` request in, `(N, 50, 8)` absolute eef chunks out.

`build_pi05_app(policy, model_name)` takes any callable on the openpi observation dict
(`observation/image`, `observation/wrist_image`, `observation/state`, `prompt`) → `(50, 8)`, so the
app is testable without openpi; `load_policy(checkpoint_dir)` builds the real one from a
checkpoint inside the patched copy (`create_trained_policy`, EMA params). The state is assembled
in the dataset's order (`contract.STATE_NAMES`): eef position, eef orientation (wxyz),
gripper_state (1 open, 0 closed) — exactly what the evaluator's `vla` mode sends.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from reanchor_sft import contract

Policy = Callable[[dict[str, Any]], np.ndarray]
CAMERA_TO_OBS = {"fixed_cam": "observation/image", "wrist_cam": "observation/wrist_image"}


class VlaRequest(BaseModel):
    type: str
    num_envs: int
    control_mode: str = "ik_abs"
    proprioception: dict[str, list[list[float]]] | None = None
    images: dict[str, list[str]] | None = None
    task_description: str | None = None
    step_ids: list[int]


def decode_image(value: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB"), np.uint8)


def assemble_state(proprio: dict[str, list[list[float]]], i: int) -> np.ndarray:
    """`contract.STATE_NAMES` order from the request's proprioception fields."""
    state = np.concatenate(
        [
            np.asarray(proprio["eef_pos"][i], np.float32),
            np.asarray(proprio["eef_orient"][i], np.float32),
            np.asarray(proprio["gripper_state"][i], np.float32),
        ]
    )
    if state.shape != (contract.STATE_DIM,):
        raise HTTPException(400, f"proprioception does not assemble to {contract.STATE_DIM}-D")
    return state


def build_pi05_app(policy: Policy, *, model_name: str) -> FastAPI:
    app = FastAPI(title="reanchor pi05 policy")

    @app.get("/info")
    def info() -> dict:
        return {
            "action_dim": contract.ACTION_DIM,
            "action_horizon": contract.ACTION_HORIZON,
            "model_name": model_name,
            "control_mode": "ik_abs",
        }

    @app.post("/predict")
    def predict(req: VlaRequest) -> dict:
        if req.type != "vla":
            raise HTTPException(400, "this server takes the vla request (images + proprioception)")
        if req.images is None or req.proprioception is None:
            raise HTTPException(400, "vla request needs images and proprioception")
        missing = set(CAMERA_TO_OBS) - set(req.images)
        if missing:
            raise HTTPException(400, f"missing camera(s): {sorted(missing)}")
        if set(req.proprioception) != {"eef_pos", "eef_orient", "gripper_state"}:
            raise HTTPException(400, "proprioception needs eef_pos, eef_orient, gripper_state")
        chunks = []
        for i in range(req.num_envs):
            obs: dict[str, Any] = {
                CAMERA_TO_OBS[cam]: decode_image(req.images[cam][i]) for cam in CAMERA_TO_OBS
            }
            obs["observation/state"] = assemble_state(req.proprioception, i)
            obs["prompt"] = req.task_description or contract.TASK_PROMPT
            chunk = np.asarray(policy(obs), dtype=np.float32)
            if chunk.shape != (contract.ACTION_HORIZON, contract.ACTION_DIM):
                raise HTTPException(500, f"policy returned {chunk.shape}, expected (50, 8)")
            chunks.append(chunk.tolist())
        return {"actions": chunks}

    @app.post("/reset")
    def reset() -> dict:
        return {"ok": True}

    return app


def load_policy(checkpoint_dir: Path, *, profile=None) -> Policy:
    """The real π0.5 policy from a milestone directory (call inside the patched copy)."""
    from openpi.policies import policy_config

    from reanchor_sft.profile import load_profile
    from reanchor_sft.schedule import SFTLaunchRequest
    from reanchor_sft.training import build_train_config, register_config

    profile = profile or load_profile()
    checkpoint_dir = Path(checkpoint_dir).resolve()
    request = SFTLaunchRequest("serve", "formal", False, 1, profile.batch_size)
    config = build_train_config(
        profile,
        request,
        dataset_repo_id=profile.dataset_repo_id,
        assets_root=checkpoint_dir / "assets",
        checkpoint_root=checkpoint_dir.parent,
        wandb_enabled=False,
    )
    register_config(config)
    policy = policy_config.create_trained_policy(config, checkpoint_dir)

    def infer(obs: dict[str, Any]) -> np.ndarray:
        return np.asarray(policy.infer(obs)["actions"])[: contract.ACTION_HORIZON]

    return infer
