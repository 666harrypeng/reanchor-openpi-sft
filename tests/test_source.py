"""`temporary_patched_openpi_copy`: a `git archive` of the pinned openpi commit with patches
0001–0003 applied, importable for the duration of a `with` block, gone afterwards. The submodule
itself is never touched. Needs git only (no openpi import)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from reanchor_sft.source import OPENPI_REVISION, OPENPI_ROOT, PATCHES, temporary_patched_openpi_copy

pytestmark = pytest.mark.skipif(
    not (OPENPI_ROOT / "pyproject.toml").exists(), reason="openpi submodule not checked out"
)


def test_copy_has_the_patches_applied():
    with temporary_patched_openpi_copy() as copy:
        assert "drop_n_last_frames" in (copy / "src/openpi/training/config.py").read_text()
        assert "active_action_dim" in (copy / "src/openpi/models/pi0_config.py").read_text()
        assert "_checkpoint_save_step" in (copy / "scripts/train.py").read_text()
        assert sys.path[0] == str(copy / "src")
    assert len(PATCHES) == 3 and OPENPI_REVISION == "15a9616a00943ada6c20a0f158e3adb39df2ccac"


def test_copy_is_removed_and_sys_path_restored_on_exit():
    before = list(sys.path)
    with temporary_patched_openpi_copy() as copy:
        path = copy
    assert not path.exists()
    assert sys.path == before


def test_wrong_revision_is_refused():
    with (
        pytest.raises(RuntimeError, match="revision"),
        temporary_patched_openpi_copy(expected_revision="0" * 40),
    ):
        pass


def test_jax_cache_literal_is_rewritten(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "jax"))
    with temporary_patched_openpi_copy() as copy:
        text = (copy / "scripts/train.py").read_text()
    assert str((tmp_path / "jax").resolve()) in text
    assert "~/.cache/jax" not in text


def test_submodule_is_left_clean():
    import subprocess

    with temporary_patched_openpi_copy():
        pass
    status = subprocess.run(
        ["git", "-C", str(OPENPI_ROOT), "status", "--porcelain"], capture_output=True, text=True
    )
    assert status.stdout == ""
