"""The dataset push CLI refuses without a token or a complete converted dataset."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from reanchor_sft import contract
from reanchor_sft.source import REPO_ROOT


def _mod():
    spec = importlib.util.spec_from_file_location(
        "push_cli", REPO_ROOT / "scripts/sft_push_dataset.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_push_checks(tmp_path: Path, monkeypatch):
    m = _mod()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        m.check(tmp_path, "a/b")
    monkeypatch.setenv("HF_TOKEN", "x")
    with pytest.raises(FileNotFoundError):
        m.check(tmp_path, "a/b")
    d = tmp_path / "a" / "b"
    d.mkdir(parents=True)
    (d / "reanchor_dataset.json").write_text(json.dumps({"episodes": 20, "frames": 3184}))
    with pytest.raises(ValueError, match="not the full dataset"):
        m.check(tmp_path, "a/b")
    (d / "reanchor_dataset.json").write_text(
        json.dumps({"episodes": contract.SOURCE_EPISODES, "frames": contract.SOURCE_FRAMES})
    )
    assert m.check(tmp_path, "a/b")["frames"] == contract.SOURCE_FRAMES
