"""`tools/openpi/thp_off.py` runs a script with transparent huge pages disabled for the process and
its children, forwarding the script's arguments unchanged."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "tools" / "openpi" / "thp_off.py"
STATUS = Path("/proc/self/status")

# Prints this process's THP_enabled flag, a child's, then its own argv.
PROBE = textwrap.dedent(
    """
    import subprocess, sys
    from pathlib import Path

    THP = ("from pathlib import Path; print(next(l for l in Path('/proc/self/status')"
           ".read_text().splitlines() if l.startswith('THP_enabled')).split()[1])")
    exec(THP)
    print(subprocess.check_output([sys.executable, "-c", THP], text=True).strip())
    print(sys.argv[1:])
    """
)


def _probe_output(tmp_path: Path, *prefix: str) -> list[str]:
    probe = tmp_path / "probe.py"
    probe.write_text(PROBE)
    cmd = [sys.executable, *prefix, str(probe), "a", "--b"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.splitlines()


@pytest.mark.skipif("THP_enabled" not in STATUS.read_text(), reason="kernel reports no THP flag")
def test_wrapper_disables_thp_for_process_and_children_and_forwards_argv(tmp_path):
    assert _probe_output(tmp_path)[:2] == ["1", "1"]
    assert _probe_output(tmp_path, str(WRAPPER)) == ["0", "0", "['a', '--b']"]
