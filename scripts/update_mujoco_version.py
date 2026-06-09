#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
VERSION_FILE = ROOT / "cmake" / "Versions.cmake"


def read_supported_version() -> str:
    text = VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(
        r'^\s*set\s*\(\s*MJ_KDL_MUJOCO_VERSION\s+"?(?P<version>[0-9]+\.[0-9]+\.[0-9]+)"?\s*\)',
        text,
        re.MULTILINE,
    )
    if not match:
        raise SystemExit(f"Could not find MJ_KDL_MUJOCO_VERSION in {VERSION_FILE}")
    return match.group("version")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def replace(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    next_text = text
    for old, new in replacements:
        next_text = next_text.replace(old, new)
    if next_text != text:
        write(path, next_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Update the supported MuJoCo version.")
    parser.add_argument("version", help="New MuJoCo version, for example 3.9.0")
    args = parser.parse_args()

    new_version = args.version.removeprefix("v")
    if not VERSION_RE.match(new_version):
        raise SystemExit(f"Invalid MuJoCo version: {args.version}")

    old_version = read_supported_version()
    old_major_minor = ".".join(old_version.split(".")[:2])
    new_major_minor = ".".join(new_version.split(".")[:2])

    replace(
        VERSION_FILE,
        [(f'MJ_KDL_MUJOCO_VERSION "{old_version}"', f'MJ_KDL_MUJOCO_VERSION "{new_version}"')],
    )

    common = [
        (old_version, new_version),
        (f"mujoco-{old_version}", f"mujoco-{new_version}"),
        (f"MuJoCo {old_major_minor}", f"MuJoCo {new_major_minor}"),
    ]
    for relpath in [
        "README.md",
        "CLAUDE.md",
        "docs/howto/urdf_to_mjcf.md",
        ".github/workflows/ci.yml",
        "pyproject.toml",
    ]:
        replace(ROOT / relpath, common)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
