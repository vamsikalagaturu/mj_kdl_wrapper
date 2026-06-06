from __future__ import annotations

import re
from pathlib import Path


def dynamic_metadata(field, settings=None, project=None):
    if field != "dependencies":
        raise ValueError(f"mujoco_dependency only supports dependencies, not {field!r}")

    version_file = Path(__file__).resolve().parents[1] / "cmake" / "MuJoCoVersion.cmake"
    text = version_file.read_text(encoding="utf-8")
    match = re.search(
        r'^\s*set\s*\(\s*MJ_KDL_MUJOCO_VERSION\s+"?(?P<version>[0-9]+\.[0-9]+\.[0-9]+)"?\s*\)',
        text,
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"Could not find MJ_KDL_MUJOCO_VERSION in {version_file}")

    return [f"mujoco=={match.group('version')}"]


if __name__ == "__main__":
    print(dynamic_metadata("dependencies")[0])
