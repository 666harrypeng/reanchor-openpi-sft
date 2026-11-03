"""Run a Python script with transparent huge pages disabled for this process and its children.

    tools/openpi/.venv/bin/python tools/openpi/thp_off.py scripts/sft_train.py ...

On multi-socket hosts with automatic NUMA balancing and THP enabled, the ~40 GB host buffers of a
checkpoint save can be migrated back and forth between NUMA nodes under compaction until the save
never completes (zero bytes written, one thread at 100 % kernel time). prctl(PR_SET_THP_DISABLE)
is inherited across fork and exec, so the trainer and its data-loader workers all run without THP.
"""

from __future__ import annotations

import ctypes
import runpy
import sys

PR_SET_THP_DISABLE = 41


def disable_thp() -> None:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_THP_DISABLE) failed")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} SCRIPT [ARGS...]")
    disable_thp()
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
