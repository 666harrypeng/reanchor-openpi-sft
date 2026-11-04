"""The SFT data contract, copied from the static dataset's own metadata (never re-derived).

Source: `data/s3/static_ikabs_200_v2.hdf5` attrs and the published LeRobot v3.0 dataset
`yypeng666/reanchor-conveyor-static-ikabs-v2` (protocol §Data). The task string is the one the
collector recorded; the LeRobot v2.1 companion dataset for openpi carries the same string.
"""

from __future__ import annotations

FPS = 25  # 1 / (sim_dt 0.01 × decimation 4)
STATE_NAMES: tuple[str, ...] = ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")
STATE_DIM = len(STATE_NAMES)  # eef position, eef orientation (wxyz), gripper
ACTION_DIM = 8  # absolute eef pose + gripper (`ik_abs`, `abs_eef_pose`, quat)
QUATERNION_DIMS = slice(3, 7)  # wxyz in state and actions: normalized by the identity (training.py)
ACTION_HORIZON = 50  # openpi default; ≥ the executor's h_exec (10 ticks)
TASK_PROMPT = "pick up the cube from the conveyor and place it into the bin"
CAMERAS: dict[str, str] = {"fixed_cam": "base_0_rgb", "wrist_cam": "left_wrist_0_rgb"}
IMAGE_HW = (480, 640)  # recorded size; openpi resizes (with padding) to 224 × 224 itself

SOURCE_HDF5 = "data/s3/static_ikabs_200_v2.hdf5"
SOURCE_HDF5_SHA256 = "6f0fd76c5526d753c09d54adb6b20597a1b65fb4816e4182826f975993410e89"
SOURCE_EPISODES = 200
SOURCE_FRAMES = 32094
DATASET_REPO_V30 = "yypeng666/reanchor-conveyor-static-ikabs-v2"  # LeRobot v3.0 (tools/lerobot)
DATASET_REPO_V21 = "yypeng666/reanchor-conveyor-static-ikabs-v2-lerobot21"  # openpi's loader
