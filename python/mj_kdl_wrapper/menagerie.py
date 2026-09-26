"""Locate (and optionally fetch) MuJoCo Menagerie models and the bundled assets.

The only model source this package provides is the official MuJoCo Menagerie
(https://github.com/google-deepmind/mujoco_menagerie). :func:`model_path` resolves in order:

1. The file named by ``env_var``, when the caller passes one and it is set.
2. ``MJ_KDL_MENAGERIE`` (a Menagerie checkout root), if set.
3. The bundled assets in the user cache, whose models derived from Menagerie's
   (``kinova_gen3/gen3.xml``) replace the upstream copy.
4. The Menagerie checkout in the user cache, populated by ``mj-kdl-fetch-menagerie``
   (or :func:`fetch`).

Other model sources (e.g. the ``robot_descriptions`` package, your own URDF/MJCF
exports) are not provided here; point ``RobotSpec.path`` or ``MJ_KDL_MENAGERIE`` at them.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path

MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
# Same commit as MJ_KDL_MENAGERIE_GIT_SHA in cmake/Versions.cmake; scripts/check_versions.py checks.
MENAGERIE_COMMIT = "4c358ef9d9d7f32ca58b40b490884a0c1726a440"

# logical name -> (Menagerie subdirectory, model file within it)
_MODELS = {
    "kinova_gen3": ("kinova_gen3", "gen3.xml"),
    "robotiq_2f85": ("robotiq_2f85", "2f85.xml"),
    "universal_robots_ur5e": ("universal_robots_ur5e", "ur5e.xml"),
    "universal_robots_ur10e": ("universal_robots_ur10e", "ur10e.xml"),
}

# Fingerprint of the bundled assets last copied into the cache.
_ASSETS_STAMP = ".mj_kdl_wrapper_assets"


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
    """Return a filesystem path to the named Menagerie model, resolved as the module says.

    An ``env_var`` that is set must name an existing file, else RuntimeError; a model found
    nowhere raises RuntimeError that says how to fetch it.
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

    # Mirrored in src/examples/example_paths.hpp:find_menagerie_model.
    env = os.environ.get("MJ_KDL_MENAGERIE")
    roots = (assets_cache_dir(), _cache_dir())
    if env:
        roots = (Path(env), *roots)
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
    """Return a filesystem path to a bundled mj_kdl_wrapper asset in the user cache.

    If ``env_var`` is set, that value is used as a user override; it must
    point at an existing file or a RuntimeError is raised. The cache is refreshed from the
    installed package when it is missing or older than it.
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

    try:
        _sync_assets()
    except RuntimeError:
        pass

    candidate = assets_cache_dir() / relative
    if candidate.exists():
        return str(candidate)

    raise RuntimeError(
        f"Asset '{relative_path}' was not found. Searched: {candidate}. Fetch bundled assets "
        f"with 'mj-kdl-fetch-menagerie' or set {env_var or 'the asset env var'} to an asset file."
    )


def _git(repo: Path, *args: str) -> None:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {result.stderr.strip()}")


def fetch(dest: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Shallow-fetch MuJoCo Menagerie at MENAGERIE_COMMIT into ``dest`` (default: cache).

    An existing checkout is kept as is; any other non-empty ``dest`` is refused. Returns a
    mapping of model name -> resolved path. Requires ``git``.
    """
    if shutil.which("git") is None:
        raise RuntimeError("git is required to fetch MuJoCo Menagerie")

    target = Path(dest) if dest is not None else _cache_dir()
    if not (target / ".git").exists():
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise RuntimeError(
                f"{target} exists and is not a Menagerie checkout; pass an empty or new --dest"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        # Fetched beside the target and moved in whole, so a failure leaves nothing behind.
        partial = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        try:
            _git(partial, "init", "-q")
            # A clone can only be shallow at a branch or tag; fetch the one pinned commit.
            _git(partial, "fetch", "-q", "--depth", "1", MENAGERIE_REPO, MENAGERIE_COMMIT)
            _git(partial, "checkout", "-q", "--detach", "FETCH_HEAD")
            partial.replace(target)
        except RuntimeError as exc:
            shutil.rmtree(partial, ignore_errors=True)
            raise RuntimeError(
                f"failed to fetch {MENAGERIE_REPO} at {MENAGERIE_COMMIT}: {exc}"
            ) from None
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise

    resolved = {}
    for name, (subdir, filename) in _MODELS.items():
        path = target / subdir / filename
        if not path.exists():
            raise RuntimeError(f"expected {path} after fetch, but it is missing")
        resolved[name] = str(path)
    fetch_assets()
    return resolved


def _fingerprint(root: Path) -> str:
    entries = sorted(
        f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}"
        for path in root.rglob("*")
        if path.is_file()
        for stat in (path.stat(),)
    )
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def _copy_assets(target: Path, only_if_stale: bool) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp = target / _ASSETS_STAMP
    with resources.as_file(resources.files("mj_kdl_wrapper") / "assets") as packaged:
        source = packaged if packaged.exists() else _repo_assets_dir()
        if not source.exists():
            raise RuntimeError("bundled mj_kdl_wrapper assets are missing")
        fingerprint = _fingerprint(source)
        if only_if_stale and stamp.exists() and stamp.read_text() == fingerprint:
            return str(target)
        shutil.copytree(source, target, dirs_exist_ok=True)
    stamp.write_text(fingerprint)
    return str(target)


def _sync_assets() -> str:
    return _copy_assets(assets_cache_dir(), only_if_stale=True)


def fetch_assets(dest: str | os.PathLike[str] | None = None) -> str:
    """Copy bundled mj_kdl_wrapper assets into ``dest`` (default: cache)."""
    return _copy_assets(Path(dest) if dest is not None else assets_cache_dir(), False)


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

    try:
        resolved = fetch(target)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("Resolved models:")
    for name, path in resolved.items():
        print(f"  {name}: {path}")
    print(f"Bundled assets: {assets_cache_dir()}")
    print()
    print("Model lookup uses this order:")
    print("  1. MJ_KDL_MENAGERIE environment variable")
    print(f"  2. bundled assets in the user cache: {assets_cache_dir()}")
    print(f"  3. user cache: {_cache_dir()}")
    print(f"Set MJ_KDL_MENAGERIE={target} to force this checkout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
