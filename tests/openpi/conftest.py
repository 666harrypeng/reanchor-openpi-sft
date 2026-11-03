"""Tests that need openpi / lerobot / JAX. They run in the tools/openpi environment:

    JAX_PLATFORMS=cpu tools/openpi/.venv/bin/python -m pytest tests/openpi -q

lerobot captures
`HF_LEROBOT_HOME` at import time, so it is pinned to a session-scoped temporary root here, before
anything imports lerobot; every test writes its datasets under that root."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_LEROBOT_HOME = Path(tempfile.mkdtemp(prefix="reanchor-sft-tests-"))
os.environ["HF_LEROBOT_HOME"] = str(_LEROBOT_HOME)
pytest.importorskip("lerobot", reason="openpi environment only (tools/openpi/sync.sh)")
pytest.importorskip("jax", reason="openpi environment only (tools/openpi/sync.sh)")
# the module that captures HF_LEROBOT_HOME: import it now, while the variable points at the root
pytest.importorskip("lerobot.common.datasets.lerobot_dataset")


@pytest.fixture(scope="session")
def lerobot_home() -> Path:
    return _LEROBOT_HOME


@pytest.fixture(scope="session")
def patched_openpi():
    """The patched copy on `sys.path` for the whole session (enter it before any openpi import)."""
    from reanchor_sft.source import temporary_patched_openpi_copy

    with temporary_patched_openpi_copy() as copy:
        yield copy


@pytest.fixture(scope="session")
def tiny_dataset(lerobot_home: Path) -> str:
    """Two episodes × 6 frames, the reader test's tiny images, the contract's task string."""
    from reanchor_sft.convert import write_lerobot_v21
    from tests.test_convert_hdf5_reader import write_fake_hdf5

    repo_id = "local/tiny_v21"
    hdf5 = write_fake_hdf5(lerobot_home / "tiny.hdf5", lengths=(6, 6))
    write_lerobot_v21(hdf5, repo_id=repo_id, root=lerobot_home, source_sha256="00" * 32)
    return repo_id
