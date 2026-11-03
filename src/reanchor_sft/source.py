"""A temporary, patched copy of the pinned openpi source.

openpi lives as an untouched shallow submodule at `third_party/openpi` (commit `OPENPI_REVISION`).
Training, norm stats and serving run against a `git archive` of that commit, extracted into a
temporary directory with `patches/openpi/000{1,2,3}` applied, and importable (`<copy>/src` at
`sys.path[0]`) for the duration of a `with` block. On exit the copy is deleted, its modules are
dropped from `sys.modules` and `sys.path` is restored, so two runs in one process never see
mixed sources. The mechanism is the one the LIBERO pipeline validated.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENPI_ROOT = REPO_ROOT / "third_party" / "openpi"
OPENPI_REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
PATCHES: tuple[Path, ...] = tuple(
    REPO_ROOT / "patches" / "openpi" / name
    for name in (
        "0001-filter-incomplete-action-chunks.patch",
        "0002-save-completed-step-checkpoints.patch",
        "0003-mask-action-tails.patch",
    )
)
# project modules that import openpi: dropped with the copy so a later copy re-imports them
_OPENPI_DEPENDENT_PREFIXES = ("reanchor_sft.data", "reanchor_sft.training", "reanchor_sft.serving")
_JAX_CACHE_LITERAL = 'str(epath.Path("~/.cache/jax").expanduser())'


@contextlib.contextmanager
def temporary_patched_openpi_copy(
    *,
    openpi_root: Path = OPENPI_ROOT,
    patch_paths: tuple[Path, ...] = PATCHES,
    expected_revision: str = OPENPI_REVISION,
) -> Iterator[Path]:
    """Yield the patched copy's root with `<copy>/src` first on `sys.path`."""
    root = Path(openpi_root).resolve()
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    if head != expected_revision:
        raise RuntimeError(
            f"openpi revision mismatch: checkout {head}, expected {expected_revision}"
        )
    if _git(root, "status", "--porcelain").stdout.strip():
        raise RuntimeError("the openpi submodule must be clean (it is never edited in place)")
    archive = subprocess.run(
        ("git", "archive", expected_revision), cwd=root, check=True, capture_output=True
    ).stdout
    modules_before = set(sys.modules)
    path_before = list(sys.path)
    with tempfile.TemporaryDirectory(prefix="reanchor-openpi-copy-") as directory:
        copied = Path(directory)
        try:
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(copied, filter="data")
            # TMPDIR may be inside this repository: stop git from treating the copy as part of it
            patch_env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(copied.parent))
            for patch in patch_paths:
                _git(copied, "apply", "--check", str(Path(patch).resolve()), env=patch_env)
                _git(copied, "apply", str(Path(patch).resolve()), env=patch_env)
            cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
            if cache_dir:
                # the pinned train.py overrides JAX's environment setting with ~/.cache/jax
                script = copied / "scripts" / "train.py"
                code = script.read_text()
                if code.count(_JAX_CACHE_LITERAL) != 1:
                    raise RuntimeError("unexpected JAX cache setup in the pinned openpi train.py")
                resolved = str(Path(cache_dir).expanduser().resolve())
                script.write_text(code.replace(_JAX_CACHE_LITERAL, repr(resolved)))
            sys.path.insert(0, str(copied / "src"))
            yield copied
        finally:
            _purge_copy_modules(copied, modules_before)
            sys.path[:] = path_before
            if "beartype" in sys.modules:
                # beartype caches hints by representation across source copies
                from beartype._check.checkcache import clear_checker_caches

                clear_checker_caches()


def _git(root: Path, *args: str, env=None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args), cwd=root, check=True, capture_output=True, text=True, env=env
    )


def _purge_copy_modules(copied: Path, modules_before: set[str]) -> None:
    owned = []
    for name, module in tuple(sys.modules.items()):
        if name in modules_before and not name.startswith(_OPENPI_DEPENDENT_PREFIXES):
            continue
        module_file = getattr(module, "__file__", None)
        try:
            locations = [module_file] if module_file else list(getattr(module, "__path__", ()))
            in_copy = any(Path(loc).resolve().is_relative_to(copied) for loc in locations)
        except (OSError, RuntimeError, TypeError):
            in_copy = False
        if in_copy or name.startswith(_OPENPI_DEPENDENT_PREFIXES):
            owned.append((name, module))
    for name, module in sorted(owned, key=lambda item: item[0].count("."), reverse=True):
        sys.modules.pop(name, None)
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, child, None) is module:
            delattr(parent, child)
