"""Push an already converted LeRobot v2.1 companion dataset to the Hub (public listing, manual
gate), with its `reanchor_dataset.json` sidecar. Needs HF_TOKEN in the environment.

    tools/openpi/.venv/bin/python scripts/sft_push_dataset.py --root data/s5/lerobot \
        --repo-id yypeng666/reanchor-conveyor-static-ikabs-v2-lerobot21
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reanchor_sft import contract  # noqa: E402
from reanchor_sft.convert import SIDECAR  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", type=Path, default=Path("data/s5/lerobot"))
    p.add_argument("--repo-id", default=contract.DATASET_REPO_V21)
    return p.parse_args(argv)


def check(root: Path, repo_id: str) -> dict:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must be set in the environment to push")
    local = Path(root) / repo_id
    sidecar = local / SIDECAR
    if not sidecar.is_file():
        raise FileNotFoundError(f"{sidecar} missing: not a converted dataset")
    side = json.loads(sidecar.read_text())
    if (side["episodes"], side["frames"]) != (contract.SOURCE_EPISODES, contract.SOURCE_FRAMES):
        raise ValueError(
            f"not the full dataset: {side['episodes']} episodes / {side['frames']} frames"
        )
    return side


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    side = check(a.root, a.repo_id)
    os.environ["HF_LEROBOT_HOME"] = str(a.root.resolve())
    from huggingface_hub import HfApi
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    local = a.root / a.repo_id
    LeRobotDataset(a.repo_id, root=local).push_to_hub(private=False)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(local / SIDECAR),
        path_in_repo=SIDECAR,
        repo_id=a.repo_id,
        repo_type="dataset",
    )
    api.update_repo_settings(repo_id=a.repo_id, repo_type="dataset", private=False, gated="manual")
    print(json.dumps({"pushed": a.repo_id, "episodes": side["episodes"], "frames": side["frames"]}))


if __name__ == "__main__":
    main()
