"""Locate MuJoCo Menagerie models used by the examples.

Resolution order for a model:
1. ``MJ_KDL_MENAGERIE`` env var (a Menagerie checkout root), if set.
2. A ``third_party/menagerie`` checkout next to the current dir or the repo.
3. The ``robot_descriptions`` package, which downloads and caches Menagerie on
   first use. Install it with the optional dependency::

       uv pip install "mj-kdl-wrapper[menagerie]"

   or pre-fetch the models with the ``mj-kdl-fetch-menagerie`` console script.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

# logical name -> (path within a Menagerie tree, robot_descriptions module)
_MODELS = {
    "kinova_gen3": ("kinova_gen3/gen3.xml", "gen3_mj_description"),
    "robotiq_2f85": ("robotiq_2f85/2f85.xml", "robotiq_2f85_mj_description"),
}


def _local_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("MJ_KDL_MENAGERIE")
    if env:
        roots.append(Path(env))
    roots.append(Path.cwd() / "third_party" / "menagerie")
    # repo root relative to this file: python/mj_kdl_wrapper/menagerie.py
    roots.append(Path(__file__).resolve().parents[2] / "third_party" / "menagerie")
    return roots


def model_path(name: str, *, env_var: str | None = None) -> str:
    """Return a filesystem path to the named Menagerie model.

    If ``env_var`` is given and set, that path is returned directly (user
    override). Otherwise a local checkout is preferred, falling back to
    ``robot_descriptions`` (which downloads on first use).
    """
    if env_var:
        override = os.environ.get(env_var)
        if override:
            return override

    try:
        rel, rd_module = _MODELS[name]
    except KeyError:
        raise KeyError(f"unknown Menagerie model '{name}'; known: {sorted(_MODELS)}") from None

    for root in _local_roots():
        candidate = root / rel
        if candidate.exists():
            return str(candidate)

    try:
        module = importlib.import_module(f"robot_descriptions.{rd_module}")
    except ImportError as exc:
        raise RuntimeError(
            f"Menagerie model '{name}' was not found locally. Either install the "
            f'fetch option with: uv pip install "mj-kdl-wrapper[menagerie]" (then run '
            f"mj-kdl-fetch-menagerie), fetch MuJoCo Menagerie into third_party/menagerie, "
            f"or point MJ_KDL_MENAGERIE at a Menagerie checkout."
        ) from exc
    return module.MJCF_PATH


def fetch() -> dict[str, str]:
    """Resolve (downloading if needed) every known model; return name -> path."""
    return {name: model_path(name) for name in _MODELS}


def main() -> int:
    import argparse

    argparse.ArgumentParser(
        description="Fetch the MuJoCo Menagerie models used by the mj_kdl_wrapper examples."
    ).parse_args()
    for name, resolved in fetch().items():
        print(f"{name}: {resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
