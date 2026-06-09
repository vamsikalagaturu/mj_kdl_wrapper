"""Locate (and optionally fetch) MuJoCo Menagerie models.

The only model source this package provides is the official MuJoCo Menagerie
(https://github.com/google-deepmind/mujoco_menagerie). Resolution order:

1. ``MJ_KDL_MENAGERIE`` env var (a Menagerie checkout root), if set.
2. A ``third_party/menagerie`` checkout next to the current dir or the repo.
3. The local cache populated by ``mj-kdl-fetch-menagerie`` (or :func:`fetch`).

Examples use this helper for the Kinova GEN3 arm. The Robotiq 2F-85 gripper used
by examples is bundled under the repository-level ``assets/`` directory, but the
Menagerie name remains available here for users that want to resolve an external
checkout explicitly.

Other model sources (e.g. the ``robot_descriptions`` package, your own URDF/MJCF
exports) are not provided here, but you can use them by pointing ``MJ_KDL_MODEL``
/ ``MJ_KDL_GRIPPER`` (or ``MJ_KDL_MENAGERIE``) at the files yourself.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"

# logical name -> (Menagerie subdirectory, model file within it)
_MODELS = {
    "kinova_gen3": ("kinova_gen3", "gen3.xml"),
    "robotiq_2f85": ("robotiq_2f85", "2f85.xml"),
}


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "mj_kdl_wrapper" / "menagerie"


def _roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("MJ_KDL_MENAGERIE")
    if env:
        roots.append(Path(env))
    roots.append(Path.cwd() / "third_party" / "menagerie")
    # repo root relative to this file: python/mj_kdl_wrapper/menagerie.py
    roots.append(Path(__file__).resolve().parents[2] / "third_party" / "menagerie")
    roots.append(_cache_dir())
    return roots


def model_path(name: str, *, env_var: str | None = None) -> str:
    """Return a filesystem path to the named Menagerie model.

    If ``env_var`` is set, that value is returned directly (user override).
    Otherwise a local Menagerie checkout or the fetched cache is used; if none
    exists, a RuntimeError explains how to fetch the models.
    """
    if env_var:
        override = os.environ.get(env_var)
        if override:
            return override

    try:
        subdir, filename = _MODELS[name]
    except KeyError:
        raise KeyError(f"unknown Menagerie model '{name}'; known: {sorted(_MODELS)}") from None

    for root in _roots():
        candidate = root / subdir / filename
        if candidate.exists():
            return str(candidate)

    raise RuntimeError(
        f"Menagerie model '{name}' was not found. Fetch the models with the "
        f"'mj-kdl-fetch-menagerie' console script, set MJ_KDL_MENAGERIE to a "
        f"MuJoCo Menagerie checkout, or set {env_var or 'the model env var'} to a model file."
    )


def _run(cmd: list[str]) -> bool:
    return subprocess.run(cmd, capture_output=True).returncode == 0


def fetch(dest: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Shallow-clone the full MuJoCo Menagerie into ``dest`` (default: cache).

    Returns a mapping of model name -> resolved path. Requires ``git``.
    """
    if shutil.which("git") is None:
        raise RuntimeError("git is required to fetch MuJoCo Menagerie")

    target = Path(dest) if dest is not None else _cache_dir()
    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(target, ignore_errors=True)
        if not _run(["git", "clone", "--depth", "1", MENAGERIE_REPO, str(target)]):
            raise RuntimeError(f"failed to clone {MENAGERIE_REPO}")

    resolved = {}
    for name, (subdir, filename) in _MODELS.items():
        path = target / subdir / filename
        if not path.exists():
            raise RuntimeError(f"expected {path} after fetch, but it is missing")
        resolved[name] = str(path)
    return resolved


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch the MuJoCo Menagerie models used by the mj_kdl_wrapper examples."
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Destination directory for the Menagerie checkout (default: user cache).",
    )
    args = parser.parse_args()
    for name, path in fetch(args.dest).items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
