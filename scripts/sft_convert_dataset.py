"""Convert the static HDF5 demonstrations to the LeRobot v2.1 companion dataset for openpi.

    tools/openpi/.venv/bin/python scripts/sft_convert_dataset.py \
        --input data/s3/static_ikabs_200_v2.hdf5 --root data/s5/lerobot \
        --out outputs/s5/convert.json

Writes `<root>/<repo-id>` plus `reanchor_dataset.json` beside the parquet files and a JSON report.
`--push` uploads to the Hub (public listing, manual gate; needs HF_TOKEN) — only on request.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reanchor_sft import contract  # noqa: E402
from reanchor_sft.convert import SIDECAR, sha256_file, write_lerobot_v21  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, default=Path(contract.SOURCE_HDF5))
    p.add_argument("--repo-id", default=contract.DATASET_REPO_V21)
    p.add_argument("--root", type=Path, default=Path("data/s5/lerobot"))
    p.add_argument("--episodes", type=int, default=None, help="convert only the first N (smoke)")
    p.add_argument(
        "--writer-threads",
        type=int,
        default=None,
        help="LeRobot image-writer threads (PNG encoding); default min(16, cpus), 0 = synchronous",
    )
    p.add_argument("--out", type=Path, required=True, help="JSON report")
    p.add_argument("--push", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["HF_LEROBOT_HOME"] = str(args.root.resolve())
    t0 = time.time()
    digest = sha256_file(args.input)
    if args.episodes is None and digest != contract.SOURCE_HDF5_SHA256:
        raise SystemExit(f"{args.input} sha256 {digest} is not the recorded source")
    out = write_lerobot_v21(
        args.input,
        repo_id=args.repo_id,
        root=args.root,
        episodes=args.episodes,
        source_sha256=digest,
        writer_threads=args.writer_threads,
    )
    side = json.loads((out / SIDECAR).read_text())
    report = {**side, "path": str(out), "seconds": round(time.time() - t0, 1)}
    if args.episodes is None and (side["episodes"], side["frames"]) != (
        contract.SOURCE_EPISODES,
        contract.SOURCE_FRAMES,
    ):
        got = f"{side['episodes']} episodes / {side['frames']} frames"
        raise SystemExit(f"converted {got}, expected the contract's full dataset")
    if args.push:
        if not os.environ.get("HF_TOKEN"):
            raise SystemExit("--push needs HF_TOKEN in the environment")
        from huggingface_hub import HfApi
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        LeRobotDataset(args.repo_id, root=out).push_to_hub(private=False)
        HfApi().upload_file(
            path_or_fileobj=str(out / SIDECAR),
            path_in_repo=SIDECAR,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        HfApi().update_repo_settings(
            repo_id=args.repo_id, repo_type="dataset", private=False, gated="manual"
        )
        report["pushed"] = args.repo_id
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
