"""Serve a π0.5 SFT checkpoint over ReflexBench's policy-server protocol (vla requests).

    tools/openpi/.venv/bin/python scripts/serve_pi05.py \
        --checkpoint <run>/checkpoints/pi05_conveyor_static/<run>/3000 --port 8003

XLA allocates on demand (no preallocation) and `--mem-fraction` caps it, so Isaac Sim and the policy
share one GPU: bf16 π0.5 holds ≈ 9 GB once loaded (measured on the 4080).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint", type=Path, required=True, help="milestone dir with params/ and assets/"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--mem-fraction", type=float, default=0.75)
    args = p.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.mem_fraction))

    import uvicorn

    from reanchor_sft.runtime import apply_project_local_env
    from reanchor_sft.serving import build_pi05_app, load_policy
    from reanchor_sft.source import temporary_patched_openpi_copy

    apply_project_local_env()
    with temporary_patched_openpi_copy():
        policy = load_policy(args.checkpoint)
        app = build_pi05_app(policy, model_name=f"pi05_conveyor_static@{args.checkpoint.name}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
