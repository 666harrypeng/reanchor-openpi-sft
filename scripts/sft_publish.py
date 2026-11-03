"""Publish SFT milestones (inference files only) to the Hub: public listing, manual gate.

    tools/openpi/.venv/bin/python scripts/sft_publish.py \
        --checkpoint-dir outputs/s5/runs/<run>/checkpoints/pi05_conveyor_static/<run> \
        --steps 1000 2000 3000 --repo <user>/reanchor-pi05-conveyor-static-v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reanchor_sft.publish import publish_checkpoints  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, nargs="+", required=True)
    p.add_argument("--repo", required=True)
    a = p.parse_args()
    receipts = sorted(a.checkpoint_dir.glob("verified_run-step*.json"))
    receipt = receipts[-1] if receipts else a.checkpoint_dir / "verified_run-missing.json"
    oid = publish_checkpoints(
        root=a.checkpoint_dir, steps=tuple(a.steps), repo_id=a.repo, receipt=receipt
    )
    print(f"published {a.repo} @ {oid}")


if __name__ == "__main__":
    main()
