"""Publish inference milestones to the Hugging Face Hub: public listing, manual gate.

Uploads only what inference needs — `<step>/params/**`, `<step>/assets/**` (norm stats) and
`<step>/_CHECKPOINT_METADATA` — plus the pi05 licences, a model card and the run receipt. Never
`train_state/`, W&B ids or temporary orbax directories. Credentials come from `HF_TOKEN` in the
environment only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from reanchor_sft import contract
from reanchor_sft.source import REPO_ROOT

LICENCE_DIR = REPO_ROOT / "licenses" / "pi05"
LICENCE_FILES = ("LICENSE", "NOTICE", "GEMMA_TERMS.txt")
TEMP_MARKER = "orbax-checkpoint-tmp"


def inference_files(root: Path, steps: tuple[int, ...]) -> dict[str, Path]:
    """Map `path_in_repo` → local file for the requested milestones; strict about completeness."""
    root = Path(root)
    files: dict[str, Path] = {}
    for step in steps:
        d = root / str(step)
        if not d.is_dir():
            raise FileNotFoundError(f"milestone {step} missing under {root}")
        for sub in ("params", "assets"):
            if not (d / sub).is_dir() or not any((d / sub).rglob("*")):
                raise FileNotFoundError(f"milestone {step}: {sub}/ missing or empty")
        meta = d / "_CHECKPOINT_METADATA"
        if not meta.is_file():
            raise FileNotFoundError(f"milestone {step}: _CHECKPOINT_METADATA missing")
        norm = list((d / "assets").rglob("norm_stats.json"))
        if len(norm) != 1:
            raise ValueError(
                f"milestone {step}: expected exactly one norm_stats.json, got {len(norm)}"
            )
        for p in [meta, *(d / "params").rglob("*"), *(d / "assets").rglob("*")]:
            if any(TEMP_MARKER in part for part in p.parts):
                raise ValueError(f"milestone {step}: temporary checkpoint directory present ({p})")
            if p.is_file():
                files[p.relative_to(root).as_posix()] = p
    return files


def _model_card(repo_id: str, steps: tuple[int, ...], receipt: dict) -> str:
    rows = "\n".join(
        f"| {k} | {receipt[k]} |"
        for k in sorted(receipt)
        if not isinstance(receipt[k], (dict, list))
    )
    return f"""---
license: other
license_name: pi05
license_link: LICENSE
tags: [robotics, pi0.5, openpi, reanchor]
---

# {repo_id}

π0.5 full-parameter SFT on the static conveyor pick-and-place demonstrations
(`{contract.DATASET_REPO_V21}`, {contract.SOURCE_EPISODES} episodes,
{contract.SOURCE_FRAMES} frames, {contract.FPS} Hz). Task string: "{contract.TASK_PROMPT}".
State and action: {", ".join(contract.STATE_NAMES)} (absolute end-effector pose + gripper,
action horizon {contract.ACTION_HORIZON}). Cameras: {contract.CAMERAS}.

Milestones: {", ".join(str(s) for s in steps)} (`<step>/params`,
`<step>/assets/**/norm_stats.json`). Load with openpi's `create_trained_policy`; the serving
contract is ReflexBench's `vla` request.

## Run receipt

| field | value |
| --- | --- |
{rows}

Licences: `LICENSE`, `NOTICE`, `GEMMA_TERMS.txt` (base model terms).
"""


def publish_checkpoints(
    *, root: Path, steps: tuple[int, ...], repo_id: str, receipt: Path, api=None
) -> str:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must be set in the environment to publish")
    files = inference_files(root, steps)
    receipt_path = Path(receipt)
    receipt_data = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    from huggingface_hub import CommitOperationAdd, HfApi

    api = api or HfApi()
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    api.update_repo_settings(repo_id, repo_type="model", private=False, gated="manual")
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in files.items()]
    ops += [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(LICENCE_DIR / name))
        for name in LICENCE_FILES
    ]
    card = _model_card(repo_id, steps, receipt_data).encode()
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card))
    if receipt_path.is_file():
        ops.append(
            CommitOperationAdd(path_in_repo=receipt_path.name, path_or_fileobj=str(receipt_path))
        )
    result = api.create_commit(
        repo_id,
        repo_type="model",
        operations=ops,
        commit_message=f"Publish SFT milestones {', '.join(str(s) for s in steps)}",
    )
    return str(result.oid)
