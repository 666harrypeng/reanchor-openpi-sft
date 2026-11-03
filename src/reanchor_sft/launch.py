"""The training launcher: one entry for smoke and formal runs, with a launch contract and receipt.

Flow: profile → launch request → schedule → config (via the backends) → norm stats present →
preflight report (`<work_dir>/preflight.json`) → stop on `--check-data-only` → launch contract
(`<checkpoint_dir>.launch-contract.json`, written once, byte-compared on `--resume`) → training
→ milestone directories verified → receipt `<checkpoint_dir>/verified_run-step<N>.json` →
optional publish (formal runs only).

openpi is reached only through `Backends`, so the launcher's logic is tested without JAX; the
real backends (`default_backends`) import `reanchor_sft.training` inside the patched copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reanchor_sft import contract
from reanchor_sft.convert import SIDECAR
from reanchor_sft.profile import PROFILE_PATH, SFTProfile, load_profile
from reanchor_sft.schedule import SFTLaunchRequest, resolve_schedule
from reanchor_sft.source import REPO_ROOT


@dataclass(frozen=True)
class LaunchArgs:
    run_name: str
    mode: str
    dataset_root: Path
    dataset_repo_id: str
    work_dir: Path
    batch_size: int | None  # None: the profile's
    device_count: int | None  # None: detected
    resume: bool
    check_data_only: bool
    publish_repo: str | None
    wandb_disabled: bool
    source_frames: int | None = None  # None: the sidecar's frame count
    profile_path: Path = PROFILE_PATH


@dataclass(frozen=True)
class Backends:
    device_count: Callable[[], int]
    build_config: Callable[..., Any]
    norm_stats: Callable[..., Path]
    preflight: Callable[[Any], dict]
    train: Callable[..., None]
    publish: Callable[..., str]


CONTRACT_KEYS = (
    "mode",
    "run_name",
    "dataset_repo_id",
    "dataset_sidecar_sha256",
    "profile_sha256",
    "batch_size",
    "device_count",
    "warmup_steps",
    "decay_steps",  # the LR clock; a smoke resume (100 → 120 updates) keeps it
    "training_parallelism",
)


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _contract_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir.parent / f"{checkpoint_dir.name}.launch-contract.json"


def run(args: LaunchArgs, *, backends: Backends) -> dict:
    profile: SFTProfile = load_profile(args.profile_path)
    sidecar_path = Path(args.dataset_root) / args.dataset_repo_id / SIDECAR
    if not sidecar_path.exists():
        raise FileNotFoundError(f"{sidecar_path}: convert the dataset first ({SIDECAR} missing)")
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("prompt") != contract.TASK_PROMPT:
        raise ValueError("dataset task string differs from the contract")
    frames = int(args.source_frames or sidecar["frames"])
    if args.publish_repo and args.mode != "formal":
        raise ValueError("publish is for formal runs only (a smoke checkpoint is not a milestone)")
    device_count = args.device_count or int(backends.device_count())
    request = SFTLaunchRequest(
        run_name=args.run_name,
        mode=args.mode,
        resume=args.resume,
        device_count=device_count,
        batch_size=args.batch_size or profile.batch_size,
    )
    schedule = resolve_schedule(profile, request)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    assets_root, checkpoint_root = work / "assets", work / "checkpoints"
    config = backends.build_config(
        profile,
        request,
        dataset_repo_id=args.dataset_repo_id,
        assets_root=assets_root,
        checkpoint_root=checkpoint_root,
        wandb_enabled=not args.wandb_disabled,
    )
    norm_stats = Path(config.assets_dirs) / args.dataset_repo_id / "norm_stats.json"
    if not norm_stats.exists():
        norm_stats = backends.norm_stats(config, expected_frames=frames)
    report: dict[str, Any] = {
        "implementation_revision": _git("rev-parse", "HEAD"),
        "implementation_dirty": bool(_git("status", "--porcelain")),
        "profile_sha256": _sha256_text(Path(args.profile_path)),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_sidecar_sha256": _sha256_text(sidecar_path),
        "source_frames": frames,
        "mode": args.mode,
        "run_name": args.run_name,
        "batch_size": schedule.batch_size,
        "device_count": device_count,
        "per_device_batch": schedule.per_device_batch,
        "num_steps": schedule.num_train_steps,
        "warmup_steps": schedule.warmup_steps,
        "decay_steps": schedule.decay_steps,
        "checkpoint_steps": list(schedule.checkpoint_steps),
        "training_examples": schedule.batch_size * schedule.num_train_steps,
        "equivalent_epochs": schedule.batch_size * schedule.num_train_steps / frames,
        "reference_training_examples": profile.training_examples,
        "training_parallelism": "replicated_data_parallel",
        "norm_stats_sha256": _sha256_text(norm_stats),
        "checkpoint_dir": str(config.checkpoint_dir),
        "training_started": False,
        "training_completed": False,
    }
    report.update(backends.preflight(config))
    (work / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.check_data_only:
        return report

    checkpoint_dir = Path(config.checkpoint_dir)
    launch_contract = {k: report[k] for k in CONTRACT_KEYS}
    contract_path = _contract_path(checkpoint_dir)
    if args.resume:
        if not contract_path.exists():
            raise FileNotFoundError(f"--resume without a launch contract at {contract_path}")
        previous = json.loads(contract_path.read_text())
        if previous != launch_contract:
            differing = sorted(k for k in CONTRACT_KEYS if previous.get(k) != launch_contract[k])
            raise RuntimeError(f"resume contract differs from the original launch: {differing}")
    else:
        if checkpoint_dir.exists() or contract_path.exists():
            raise FileExistsError(f"{checkpoint_dir} exists; use --resume or a new run name")
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        with open(contract_path, "x") as fh:
            json.dump(launch_contract, fh, indent=2)

    report["training_started"] = True
    t0 = time.time()
    backends.train(config, device_count=device_count)
    report["elapsed_seconds"] = round(time.time() - t0, 1)
    for step in schedule.checkpoint_steps:
        d = checkpoint_dir / str(step)
        if not ((d / "params").is_dir() and (d / "assets").is_dir()):
            raise RuntimeError(f"milestone {step} incomplete under {checkpoint_dir}")
    report["training_completed"] = True
    receipt = checkpoint_dir / f"verified_run-step{schedule.num_train_steps}.json"
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    report["receipt"] = str(receipt)
    if args.publish_repo:
        report["published_revision"] = backends.publish(
            root=checkpoint_dir,
            steps=schedule.checkpoint_steps,
            repo_id=args.publish_repo,
            receipt=receipt,
        )
    return report


# ----------------------------------------------------------------------------- real backends
def default_backends() -> Backends:
    """openpi-backed implementations; call inside `temporary_patched_openpi_copy`."""
    from reanchor_sft import training

    def device_count() -> int:
        import jax

        return jax.device_count()

    def preflight(config: Any) -> dict:
        import jax
        from openpi.training import data_loader as dl

        data_config = config.data.create(config.assets_dirs, config.model)
        ds = dl.create_torch_dataset(data_config, config.model.action_horizon, config.model)
        ds = dl.TransformedDataset(
            ds,
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                *data_config.model_transforms.inputs,
            ],
        )
        item = ds[0]
        mask = item["action_loss_mask"]
        return {
            "dataset_frames": len(ds),
            "action_shape": list(item["actions"].shape),
            "action_mask_tail_false": bool(not mask[:, contract.ACTION_DIM :].any()),
            "prompt_tokens": bool(item["tokenized_prompt"].any()),
            "devices": [str(d) for d in jax.devices()],
            "device_kind": jax.devices()[0].device_kind,
        }

    def publish(**kw) -> str:
        from reanchor_sft.publish import publish_checkpoints

        return publish_checkpoints(**kw)

    return Backends(
        device_count=device_count,
        build_config=training.build_train_config,
        norm_stats=training.compute_norm_stats,
        preflight=preflight,
        train=training.run_training,
        publish=publish,
    )


def main_from_cli(args: LaunchArgs) -> dict:
    """Entry for scripts/sft_train.py: pins the environment, enters the patched copy, runs."""
    from reanchor_sft.runtime import apply_project_local_env
    from reanchor_sft.source import temporary_patched_openpi_copy

    apply_project_local_env()
    os.environ["HF_LEROBOT_HOME"] = str(Path(args.dataset_root).resolve())
    with temporary_patched_openpi_copy():
        report = run(args, backends=default_backends())
    print(json.dumps({k: v for k, v in report.items() if k != "devices"}, indent=2))
    return report


def launch_args_dict(args: LaunchArgs) -> dict:
    return asdict(args)
