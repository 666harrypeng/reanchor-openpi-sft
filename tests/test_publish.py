"""The milestone publisher: inference files only, public listing with a manual gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reanchor_sft.publish import inference_files, publish_checkpoints


def _ckpt(root: Path, steps=(1000, 2000, 3000), broken=()):
    for s in steps:
        d = root / str(s)
        (d / "params" / "ocdbt.process_0").mkdir(parents=True)
        (d / "params" / "ocdbt.process_0" / "data").write_bytes(b"w" * 16)
        (d / "params" / "_METADATA").write_text("{}")
        (d / "assets" / "yypeng666" / "ds").mkdir(parents=True)
        (d / "assets" / "yypeng666" / "ds" / "norm_stats.json").write_text("{}")
        (d / "train_state").mkdir()
        (d / "train_state" / "opt").write_bytes(b"o" * 8)
        (d / "_CHECKPOINT_METADATA").write_text("{}")
    for s in broken:
        (root / str(s) / "params" / "orbax-checkpoint-tmp-1").mkdir(parents=True, exist_ok=True)
    (root / "wandb_id.txt").write_text("abc")
    (root / "verified_run-step3000.json").write_text(json.dumps({"num_steps": 3000}))
    return root


def test_allowlist_excludes_train_state_logs_and_temp_dirs(tmp_path: Path):
    root = _ckpt(tmp_path / "run")
    files = inference_files(root, (1000, 3000))
    keys = sorted(files)
    assert keys == [
        "1000/_CHECKPOINT_METADATA",
        "1000/assets/yypeng666/ds/norm_stats.json",
        "1000/params/_METADATA",
        "1000/params/ocdbt.process_0/data",
        "3000/_CHECKPOINT_METADATA",
        "3000/assets/yypeng666/ds/norm_stats.json",
        "3000/params/_METADATA",
        "3000/params/ocdbt.process_0/data",
    ]
    with pytest.raises(FileNotFoundError, match="2500"):
        inference_files(root, (2500,))
    _ckpt(tmp_path / "run2", broken=(1000,))
    with pytest.raises(ValueError, match="orbax-checkpoint-tmp"):
        inference_files(tmp_path / "run2", (1000,))


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):
        self.calls.append(("create_repo", repo_id, repo_type, private, exist_ok))

    def update_repo_settings(self, repo_id, *, repo_type, private, gated):
        self.calls.append(("update_repo_settings", repo_id, repo_type, private, gated))

    def create_commit(self, repo_id, *, repo_type, operations, commit_message):
        self.calls.append(
            ("create_commit", repo_id, repo_type, sorted(o.path_in_repo for o in operations))
        )

        class R:
            oid = "deadbeef"

        return R()


def test_publish_sets_public_and_manual_gate_and_uploads_licences(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    root = _ckpt(tmp_path / "run")
    api = FakeApi()
    oid = publish_checkpoints(
        root=root,
        steps=(1000, 2000, 3000),
        repo_id="yypeng666/reanchor-pi05-test",
        receipt=root / "verified_run-step3000.json",
        api=api,
    )
    assert oid == "deadbeef"
    kinds = [c[0] for c in api.calls]
    assert kinds == ["create_repo", "update_repo_settings", "create_commit"]
    assert api.calls[0][3] is False  # public
    assert api.calls[1][4] == "manual" and api.calls[1][3] is False
    paths = api.calls[2][3]
    assert "README.md" in paths and "LICENSE" in paths and "GEMMA_TERMS.txt" in paths
    assert "NOTICE" in paths and "verified_run-step3000.json" in paths
    assert "3000/params/ocdbt.process_0/data" in paths and not any(
        "train_state" in p for p in paths
    )
    assert not any("wandb" in p for p in paths)


def test_publish_requires_a_token(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    root = _ckpt(tmp_path / "run")
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        publish_checkpoints(
            root=root, steps=(1000,), repo_id="a/b", receipt=root / "x", api=FakeApi()
        )
