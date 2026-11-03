"""HDF5 demonstrations → LeRobot v2.1 dataset for openpi's loader.

The published dataset (`contract.DATASET_REPO_V30`) is LeRobot v3.0, which openpi's pinned
lerobot cannot read. This module writes a companion v2.1 dataset from the same raw HDF5 with the
same contract: `image` (fixed_cam), `wrist_image` (wrist_cam), `state` (8), `actions` (8), one
task string per frame, 25 fps, PNG frames in parquet (`use_videos=False`). The reader part needs
h5py only; the writer imports lerobot lazily (tools/openpi environment).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

from reanchor_sft import contract
from reanchor_sft.source import OPENPI_REVISION

LEROBOT_REVISION = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"  # openpi's pin
SIDECAR = "reanchor_dataset.json"


@dataclass(frozen=True)
class Header:
    prompt: str
    fps: float
    control_mode: str
    action_format: str
    orientation_rep: str
    episodes: int


@dataclass(frozen=True)
class Episode:
    name: str
    state: (
        np.ndarray
    )  # (T, 8) float32: eef_pos, eef_orient (wxyz), gripper_state (1 open, 0 closed)
    actions: np.ndarray  # (T, 8) float32: absolute eef pose + gripper command (+1 open, −1 close)
    fixed_cam: np.ndarray  # (T, H, W, 3) uint8
    wrist_cam: np.ndarray


def _attr(f: h5py.File, key: str):
    v = f.attrs[key]
    return v.decode() if isinstance(v, bytes) else v


def _demo_order(f: h5py.File) -> list[str]:
    return sorted(f["data"].keys(), key=lambda d: int(d.split("_")[-1]))


def read_header(hdf5: Path) -> Header:
    with h5py.File(hdf5, "r") as f:
        return Header(
            prompt=str(_attr(f, "prompt")),
            fps=float(_attr(f, "fps")),
            control_mode=str(_attr(f, "control_mode")),
            action_format=str(_attr(f, "action_format")),
            orientation_rep=str(_attr(f, "orientation_rep")),
            episodes=len(f["data"].keys()),
        )


def check_header(h: Header) -> None:
    if h.prompt != contract.TASK_PROMPT:
        raise ValueError(f"prompt {h.prompt!r} is not the contract's {contract.TASK_PROMPT!r}")
    if h.fps != contract.FPS:
        raise ValueError(f"fps {h.fps} is not the contract's {contract.FPS}")
    if (h.control_mode, h.action_format, h.orientation_rep) != ("ik_abs", "abs_eef_pose", "quat"):
        raise ValueError(f"unexpected action identity: {h}")


def iter_episodes(hdf5: Path, limit: int | None = None) -> Iterator[Episode]:
    """Episodes in index order; every demo must be a success (the collector keeps only those)."""
    check_header(read_header(hdf5))
    with h5py.File(hdf5, "r") as f:
        names = _demo_order(f)[: limit if limit is not None else None]
        for name in names:
            d = f["data"][name]
            if not bool(d.attrs.get("success", False)):
                raise ValueError(
                    f"{name}: success attribute is not set; the dataset keeps successes only"
                )
            state = np.concatenate(
                [d["obs/eef_pos"][:], d["obs/eef_orient"][:], d["obs/gripper_state"][:]], axis=1
            ).astype(np.float32)
            actions = d["actions"][:].astype(np.float32)
            if state.shape[1] != contract.STATE_DIM or actions.shape[1] != contract.ACTION_DIM:
                raise ValueError(f"{name}: state {state.shape} / actions {actions.shape}")
            yield Episode(
                name=name,
                state=state,
                actions=actions,
                fixed_cam=d["obs/fixed_cam_rgb"][:],
                wrist_cam=d["obs/wrist_cam_rgb"][:],
            )


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _features(hw: tuple[int, int]) -> dict:
    img = {"dtype": "image", "shape": (hw[0], hw[1], 3), "names": ["height", "width", "channel"]}
    vec = {"dtype": "float32", "shape": (contract.STATE_DIM,), "names": list(contract.STATE_NAMES)}
    return {"image": img, "wrist_image": img, "state": vec, "actions": dict(vec)}


def write_lerobot_v21(
    hdf5: Path,
    *,
    repo_id: str,
    root: Path,
    episodes: int | None = None,
    source_sha256: str | None = None,
) -> Path:
    """Write `<root>/<repo_id>` (staged under a `.building` name, renamed when complete)."""
    import os

    os.environ.setdefault("HF_LEROBOT_HOME", str(Path(root).resolve()))
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    final = Path(root) / repo_id
    if final.exists():
        raise FileExistsError(f"{final} exists; remove it to re-convert")
    staging = final.parent / f".{final.name}.building"
    if staging.exists():
        import shutil

        shutil.rmtree(staging)
    header = read_header(hdf5)
    check_header(header)
    first = next(iter_episodes(hdf5, limit=1))
    hw = first.fixed_cam.shape[1:3]
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        root=staging,
        robot_type="franka",
        fps=contract.FPS,
        features=_features(hw),
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    n_eps = n_frames = 0
    for ep in iter_episodes(hdf5, limit=episodes):
        for t in range(len(ep.state)):
            ds.add_frame(
                {
                    "image": ep.fixed_cam[t],
                    "wrist_image": ep.wrist_cam[t],
                    "state": ep.state[t],
                    "actions": ep.actions[t],
                    "task": header.prompt,
                }
            )
        ds.save_episode()
        ds.hf_dataset = ds.create_hf_dataset()  # free the in-memory table between episodes
        n_eps += 1
        n_frames += len(ep.state)
    sidecar = {
        "format_id": "reanchor_lerobot_v21_v1",
        "repo_id": repo_id,
        "source_hdf5": str(hdf5),
        "source_sha256": source_sha256,
        "episodes": n_eps,
        "frames": n_frames,
        "fps": contract.FPS,
        "prompt": header.prompt,
        "image_hw": list(hw),
        "state_names": list(contract.STATE_NAMES),
        "action_names": list(contract.STATE_NAMES),
        "openpi_revision": OPENPI_REVISION,
        "lerobot_revision": LEROBOT_REVISION,
        "header": asdict(header),
    }
    (staging / SIDECAR).write_text(json.dumps(sidecar, indent=2) + "\n")
    staging.rename(final)
    return final
