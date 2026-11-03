"""The training launcher: preflight, launch contract, resume rules and the run receipt. openpi is
replaced by injected backends."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from reanchor_sft import contract
from reanchor_sft.launch import Backends, LaunchArgs, run


@dataclass
class FakeConfig:
    name: str = "pi05_conveyor_static"
    exp_name: str = ""
    batch_size: int = 0
    num_train_steps: int = 0
    keep_period: int = 0
    save_interval: int = 0
    checkpoint_dir: Path = Path()
    assets_dirs: Path = Path()
    policy_metadata: dict = field(default_factory=dict)


@dataclass
class Fakes:
    """Records what the launcher asked for; `train` writes the milestone directories."""

    trained: list = field(default_factory=list)
    norm_stats_calls: int = 0
    fail_training: bool = False

    def device_count(self) -> int:
        return 1

    def build_config(self, profile, request, **kw) -> FakeConfig:
        from reanchor_sft.schedule import resolve_schedule

        s = resolve_schedule(profile, request)
        return FakeConfig(
            exp_name=request.run_name,
            batch_size=s.batch_size,
            num_train_steps=s.num_train_steps,
            keep_period=s.milestone_interval,
            save_interval=s.rolling_save_interval,
            checkpoint_dir=Path(kw["checkpoint_root"]) / "pi05_conveyor_static" / request.run_name,
            assets_dirs=Path(kw["assets_root"]) / "pi05_conveyor_static",
            policy_metadata={"checkpoint_steps": list(s.checkpoint_steps)},
        )

    def norm_stats(self, config, *, expected_frames) -> Path:
        self.norm_stats_calls += 1
        out = config.assets_dirs / "local/tiny" / "norm_stats.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('{"norm_stats": {}}')
        return out

    def preflight(self, config) -> dict:
        return {"action_batch_shape": [8, 50, 32], "prompt_tokens": True}

    def train(self, config, *, device_count) -> None:
        self.trained.append((config.exp_name, config.batch_size, config.num_train_steps))
        if self.fail_training:
            raise RuntimeError("boom")
        for step in config.policy_metadata["checkpoint_steps"]:
            for sub in ("params", "assets", "train_state"):
                (config.checkpoint_dir / str(step) / sub).mkdir(parents=True, exist_ok=True)
            (config.checkpoint_dir / str(step) / "_CHECKPOINT_METADATA").write_text("{}")


def _args(tmp_path: Path, **over) -> LaunchArgs:
    base = dict(
        run_name="smoke-1",
        mode="smoke",
        dataset_root=tmp_path / "lerobot",
        dataset_repo_id="local/tiny",
        work_dir=tmp_path / "work",
        batch_size=8,
        device_count=None,
        resume=False,
        check_data_only=False,
        publish_repo=None,
        wandb_disabled=True,
        source_frames=12,
    )
    base.update(over)
    return LaunchArgs(**base)


def _backends(f: Fakes) -> Backends:
    return Backends(
        device_count=f.device_count,
        build_config=f.build_config,
        norm_stats=f.norm_stats,
        preflight=f.preflight,
        train=f.train,
        publish=lambda **kw: "oid",
    )


def _sidecar(tmp_path: Path) -> None:
    d = tmp_path / "lerobot" / "local" / "tiny"
    d.mkdir(parents=True, exist_ok=True)
    (d / "reanchor_dataset.json").write_text(
        json.dumps({"frames": 12, "prompt": contract.TASK_PROMPT})
    )


def test_check_data_only_writes_preflight_and_does_not_train(tmp_path: Path):
    _sidecar(tmp_path)
    f = Fakes()
    report = run(_args(tmp_path, check_data_only=True), backends=_backends(f))
    pre = json.loads((tmp_path / "work" / "preflight.json").read_text())
    assert pre["action_batch_shape"] == [8, 50, 32] and pre["batch_size"] == 8
    assert pre["device_count"] == 1 and pre["per_device_batch"] == 8
    assert f.trained == [] and f.norm_stats_calls == 1 and report["training_started"] is False


def test_contract_written_then_resume_requires_identical_contract(tmp_path: Path):
    _sidecar(tmp_path)
    f = Fakes()
    report = run(_args(tmp_path), backends=_backends(f))
    ckpt = Path(report["checkpoint_dir"])
    contract_path = ckpt.parent / f"{ckpt.name}.launch-contract.json"
    c = json.loads(contract_path.read_text())
    assert (c["mode"], c["batch_size"], c["device_count"], c["decay_steps"]) == (
        "smoke",
        8,
        1,
        96000,
    )
    assert "checkpoint_steps" not in c  # a smoke resume extends 100 → 120 by design
    assert c["training_parallelism"] == "replicated_data_parallel"
    assert f.trained == [("smoke-1", 8, 100)]
    with pytest.raises(FileExistsError):  # a second launch without --resume
        run(_args(tmp_path), backends=_backends(f))
    with pytest.raises(RuntimeError, match="batch_size"):  # resume with a different topology
        run(_args(tmp_path, resume=True, batch_size=4), backends=_backends(f))
    report2 = run(_args(tmp_path, resume=True), backends=_backends(f))  # resume: 120 steps
    assert f.trained[-1] == ("smoke-1", 8, 120) and report2["checkpoint_steps"] == [100, 120]


def test_receipt_fields_and_equivalent_epochs(tmp_path: Path):
    _sidecar(tmp_path)
    f = Fakes()
    report = run(
        _args(tmp_path, mode="formal", run_name="formal-1", batch_size=8), backends=_backends(f)
    )
    ckpt = Path(report["checkpoint_dir"])
    receipt = json.loads((ckpt / "verified_run-step96000.json").read_text())
    assert receipt["num_steps"] == 96000 and receipt["batch_size"] == 8
    assert receipt["training_examples"] == 768_000
    assert receipt["equivalent_epochs"] == pytest.approx(768_000 / 12)
    assert receipt["checkpoint_steps"] == [32000, 64000, 96000]
    assert receipt["training_completed"] is True and "elapsed_seconds" in receipt
    assert receipt["implementation_revision"] and "profile_sha256" in receipt
    assert receipt["dataset_repo_id"] == "local/tiny" and receipt["per_device_batch"] == 8


def test_smoke_refuses_publish_and_missing_sidecar_is_an_error(tmp_path: Path):
    _sidecar(tmp_path)
    with pytest.raises(ValueError, match="publish"):
        run(_args(tmp_path, publish_repo="x/y"), backends=_backends(Fakes()))
    with pytest.raises(FileNotFoundError, match="reanchor_dataset.json"):
        run(_args(tmp_path, dataset_repo_id="local/absent"), backends=_backends(Fakes()))


def test_failed_training_leaves_no_receipt(tmp_path: Path):
    _sidecar(tmp_path)
    f = Fakes(fail_training=True)
    with pytest.raises(RuntimeError, match="boom"):
        run(_args(tmp_path), backends=_backends(f))
    assert not list((tmp_path / "work").rglob("verified_run-*.json"))
