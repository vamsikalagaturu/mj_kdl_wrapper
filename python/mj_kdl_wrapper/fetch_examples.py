"""Copy the bundled mj_kdl_wrapper example scripts to a local directory.

The example scripts are shipped inside the wheel under
``mj_kdl_wrapper/examples``. This helper copies them out as a sibling
``examples/`` directory that you can run from directly. The scripts resolve
Menagerie models and bundled assets (the Robotiq 2F-85 gripper, table, and mug
MJCF files) via :mod:`mj_kdl_wrapper.menagerie`, which reads from the user
cache populated by ``mj-kdl-fetch-menagerie`` -- not from the current working
directory.
"""

from __future__ import annotations

import argparse
import shutil
from importlib import resources
from pathlib import Path


def copy_examples(dest: Path) -> None:
    """Copy the bundled example scripts into ``dest`` as a sibling dir."""
    pkg_root = resources.files("mj_kdl_wrapper")
    with resources.as_file(pkg_root / "examples") as examples_src:
        shutil.copytree(examples_src, dest / "examples", dirs_exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy the bundled mj_kdl_wrapper example scripts to a local directory."
    )
    parser.add_argument(
        "--dest",
        default="mj_kdl_wrapper_examples",
        help="Destination directory (default: ./mj_kdl_wrapper_examples)",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    copy_examples(dest)

    print(f"Copied examples to {dest}/")
    print("Run, e.g.:")
    print(f"  cd {dest}")
    print("  mj-kdl-fetch-menagerie")
    print("  python examples/ex_pick.py --gui")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
