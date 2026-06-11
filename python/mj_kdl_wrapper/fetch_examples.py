"""Copy the bundled mj_kdl_wrapper examples and assets to a local directory.

The examples and their assets (the Robotiq 2F-85 gripper, table, and mug MJCF
files) are shipped inside the wheel under ``mj_kdl_wrapper/examples`` and
``mj_kdl_wrapper/assets``. The example scripts resolve asset paths relative to
the current working directory, so this helper copies both out as sibling
``examples/`` and ``assets/`` directories that you can run from directly.
"""

from __future__ import annotations

import argparse
import shutil
from importlib import resources
from pathlib import Path


def copy_examples(dest: Path) -> None:
    """Copy the bundled examples and assets into ``dest`` as sibling dirs."""
    pkg_root = resources.files("mj_kdl_wrapper")
    with resources.as_file(pkg_root / "examples") as examples_src:
        shutil.copytree(examples_src, dest / "examples", dirs_exist_ok=True)
    with resources.as_file(pkg_root / "assets") as assets_src:
        shutil.copytree(assets_src, dest / "assets", dirs_exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy the bundled mj_kdl_wrapper examples and assets to a local directory."
    )
    parser.add_argument(
        "--dest",
        default="mj_kdl_wrapper_examples",
        help="Destination directory (default: ./mj_kdl_wrapper_examples)",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    copy_examples(dest)

    print(f"Copied examples and assets to {dest}/")
    print("Run, e.g.:")
    print(f"  cd {dest}")
    print("  mj-kdl-fetch-menagerie")
    print("  python examples/ex_pick.py --gui")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
