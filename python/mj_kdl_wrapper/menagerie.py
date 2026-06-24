"""Locate (and optionally fetch) MuJoCo Menagerie models.

The only model source this package provides is the official MuJoCo Menagerie
(https://github.com/google-deepmind/mujoco_menagerie). Resolution order:

1. ``MJ_KDL_MENAGERIE`` env var (a Menagerie checkout root), if set.
2. The local cache populated by ``mj-kdl-fetch-menagerie`` (or :func:`fetch`).

Examples use this helper for the Kinova GEN3 arm and bundled assets. Both are
resolved from explicit env overrides or the user cache.

Other model sources (e.g. the ``robot_descriptions`` package, your own URDF/MJCF
exports) are not provided here, but you can use them by pointing ``MJ_KDL_MODEL``
/ ``MJ_KDL_GRIPPER`` (or ``MJ_KDL_MENAGERIE``) at the files yourself.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from importlib import resources
from pathlib import Path

MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"

# logical name -> (Menagerie subdirectory, model file within it)
_MODELS = {
    "kinova_gen3": ("kinova_gen3", "gen3.xml"),
    "robotiq_2f85": ("robotiq_2f85", "2f85.xml"),
}


def _cache_root() -> Path:
    # Mirrored in src/examples/example_paths.hpp:cache_root and
    # CMakeLists.txt's _MJ_KDL_CACHE_ROOT; keep all three in sync.
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "mj_kdl_wrapper"


def _cache_dir() -> Path:
    return _cache_root() / "menagerie"


def assets_cache_dir() -> Path:
    return _cache_root() / "assets"


def _repo_assets_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "assets"


def model_path(name: str, *, env_var: str | None = None) -> str:
    """Return a filesystem path to the named Menagerie model.

    If ``env_var`` is set, that value is used as a user override; it must
    point at an existing file or a RuntimeError is raised. Otherwise the
    fetched cache is used; if it is missing, a RuntimeError explains how to
    fetch the models.
    """
    if env_var:
        override = os.environ.get(env_var)
        if override:
            if not Path(override).exists():
                raise RuntimeError(f"{env_var}={override} was set but does not exist")
            return override

    try:
        subdir, filename = _MODELS[name]
    except KeyError:
        raise KeyError(f"unknown Menagerie model '{name}'; known: {sorted(_MODELS)}") from None

    env = os.environ.get("MJ_KDL_MENAGERIE")
    roots = (Path(env), _cache_dir()) if env else (_cache_dir(),)
    for root in roots:
        candidate = root / subdir / filename
        if candidate.exists():
            return str(candidate)

    searched = ", ".join(str(root / subdir / filename) for root in roots)
    raise RuntimeError(
        f"Menagerie model '{name}' was not found. Searched: {searched}. Fetch the models "
        f"with the 'mj-kdl-fetch-menagerie' console script, set MJ_KDL_MENAGERIE to a "
        f"MuJoCo Menagerie checkout, or set {env_var or 'the model env var'} to a model file."
    )


def asset_path(relative_path: str, *, env_var: str | None = None) -> str:
    """Return a filesystem path to a bundled mj_kdl_wrapper asset.

    If ``env_var`` is set, that value is used as a user override; it must
    point at an existing file or a RuntimeError is raised.
    """
    if env_var:
        override = os.environ.get(env_var)
        if override:
            if not Path(override).exists():
                raise RuntimeError(f"{env_var}={override} was set but does not exist")
            return override

    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"asset path must be relative inside assets/: {relative_path}")

    candidate = assets_cache_dir() / relative
    if candidate.exists():
        return str(candidate)

    raise RuntimeError(
        f"Asset '{relative_path}' was not found. Searched: {candidate}. Fetch bundled assets "
        f"with 'mj-kdl-fetch-menagerie' or set {env_var or 'the asset env var'} to an asset file."
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
    fetch_assets()
    return resolved


def fetch_assets(dest: str | os.PathLike[str] | None = None) -> str:
    """Copy bundled mj_kdl_wrapper assets into ``dest`` (default: cache)."""
    target = Path(dest) if dest is not None else assets_cache_dir()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with resources.as_file(resources.files("mj_kdl_wrapper") / "assets") as assets_src:
            if assets_src.exists():
                shutil.copytree(assets_src, target, dirs_exist_ok=True)
                return str(target)
    except ImportError:
        pass
    if _repo_assets_dir().exists():
        shutil.copytree(_repo_assets_dir(), target, dirs_exist_ok=True)
        return str(target)
    raise RuntimeError("bundled mj_kdl_wrapper assets are missing")


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

    target = Path(args.dest) if args.dest is not None else _cache_dir()
    if (target / ".git").exists():
        print(f"Using existing MuJoCo Menagerie checkout: {target}")
    else:
        print(f"Fetching MuJoCo Menagerie from {MENAGERIE_REPO}")
        print(f"Destination: {target}")

    resolved = fetch(target)

    print("Resolved models:")
    for name, path in resolved.items():
        print(f"  {name}: {path}")
    print(f"Bundled assets: {assets_cache_dir()}")
    print()
    print("Model lookup uses this order:")
    print("  1. MJ_KDL_MENAGERIE environment variable")
    print(f"  2. user cache: {_cache_dir()}")
    print(f"Bundled asset lookup also uses user cache: {assets_cache_dir()}")
    print(f"Set MJ_KDL_MENAGERIE={target} to force this checkout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
