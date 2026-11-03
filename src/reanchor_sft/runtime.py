"""Project-local caches: nothing this repository downloads lands in the user's home."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_LAYOUT = {
    "HF_HOME": ".cache/huggingface",
    "OPENPI_DATA_HOME": ".cache/openpi",  # openpi's download cache (base checkpoints)
    "JAX_COMPILATION_CACHE_DIR": ".cache/jax",
    "WANDB_DIR": ".cache/wandb",
    "TMPDIR": ".cache/tmp",
}


def project_local_env(root: Path = REPO_ROOT) -> dict[str, str]:
    env: dict[str, str] = {}
    for key, rel in _LAYOUT.items():
        path = Path(root) / rel
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env


def apply_project_local_env(root: Path = REPO_ROOT) -> None:
    for key, value in project_local_env(root).items():
        os.environ.setdefault(key, value)
