"""π0.5 SFT launcher (tools/openpi environment): norm stats, preflight, launch contract, training,
receipt, optional publish. The global batch is configuration; the device count is detected.

    tools/openpi/.venv/bin/python scripts/sft_train.py --run-name smoke-pro6000 \
        --work-dir outputs/s5/runs/smoke-pro6000 --mode smoke --batch-size 32 --wandb-disabled
    tools/openpi/.venv/bin/python scripts/sft_train.py --run-name formal-h100x8 \
        --work-dir outputs/s5/runs/formal-h100x8 --mode formal --batch-size 256 \
        --publish-repo <user>/reanchor-pi05-conveyor-static-v1

`--check-data-only` stops after the preflight report; `--resume` continues a run whose launch
contract is identical (a smoke resume extends 100 → 120 updates).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reanchor_sft import contract  # noqa: E402
from reanchor_sft.launch import LaunchArgs, main_from_cli  # noqa: E402
from reanchor_sft.profile import PROFILE_PATH  # noqa: E402


def parse_args(argv: list[str] | None = None) -> LaunchArgs:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-name", required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["smoke", "formal"], default="formal")
    p.add_argument("--profile", type=Path, default=PROFILE_PATH)
    p.add_argument("--dataset-root", type=Path, default=Path("data/s5/lerobot"))
    p.add_argument("--dataset-repo-id", default=contract.DATASET_REPO_V21)
    p.add_argument("--batch-size", type=int, default=None, help="global batch (profile: 256)")
    p.add_argument("--device-count", type=int, default=None, help="default: jax.device_count()")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check-data-only", action="store_true")
    p.add_argument("--publish-repo", default=None)
    p.add_argument("--wandb-disabled", action="store_true")
    a = p.parse_args(argv)
    return LaunchArgs(
        run_name=a.run_name,
        mode=a.mode,
        dataset_root=a.dataset_root,
        dataset_repo_id=a.dataset_repo_id,
        work_dir=a.work_dir,
        batch_size=a.batch_size,
        device_count=a.device_count,
        resume=a.resume,
        check_data_only=a.check_data_only,
        publish_repo=a.publish_repo,
        wandb_disabled=a.wandb_disabled,
        profile_path=a.profile,
    )


if __name__ == "__main__":
    main_from_cli(parse_args())
