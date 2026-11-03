#!/usr/bin/env bash
# Create the openpi training / serving environment at tools/openpi/.venv from the openpi submodule's
# frozen lock (JAX 0.5.3 cuda12, flax 0.10.2, torch 2.7.1, lerobot @ openpi's pin), then add the few
# packages this repository's SFT scripts need on top. Separate from any simulator environment on purpose:
# the two dependency sets (lerobot, numpy, torch builds) must never meet.
#   bash tools/openpi/sync.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$ROOT/tools/openpi/.venv"
export UV_PROJECT_ENVIRONMENT="$VENV" GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE="${UV_LINK_MODE:-copy}"
uv sync --frozen --no-dev --python 3.11 --project "$ROOT/third_party/openpi"
uv pip install --python "$VENV/bin/python" \
  "fastapi==0.115.7" "uvicorn==0.29.0" "httpx>=0.27" "h5py>=3.10" "pytest==8.4.2" "pyyaml>=6"
"$VENV/bin/python" - <<'PY'
import jax, lerobot, openpi  # noqa: F401
print("openpi environment ready:", "jax", jax.__version__, "lerobot", lerobot.__version__, jax.devices())
PY
