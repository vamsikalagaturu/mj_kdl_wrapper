#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import urllib.request
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


def release_sha256(version: str) -> str:
    url = (
        "https://github.com/google-deepmind/mujoco/releases/download/"
        f"{version}/mujoco-{version}-linux-x86_64.tar.gz"
    )
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if new_version != old_version:
        text = VERSION_FILE.read_text(encoding="utf-8")
        write(
            VERSION_FILE,
            re.sub(
                r'(MJ_KDL_MUJOCO_SHA256 ")[0-9a-f]*(")',
                rf"\g<1>{release_sha256(new_version)}\g<2>",
                text,
            ),
        )

    common = [
        (old_version, new_version),
        (f"mujoco-{old_version}", f"mujoco-{new_version}"),
        (f"MuJoCo {old_major_minor}", f"MuJoCo {new_major_minor}"),
    ]
    paths = [
        ROOT / "README.md",
        ROOT / "CLAUDE.md",
        ROOT / "docs/howto/urdf_to_mjcf.md",
        ROOT / "pyproject.toml",
    ]
    # Globbed: the hardcoded ci.yml went stale when it was split into five workflows.
    paths += sorted((ROOT / ".github/workflows").glob("*.yml"))
    for path in paths:
        replace(path, common)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
