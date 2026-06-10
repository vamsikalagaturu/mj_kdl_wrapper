#!/usr/bin/env python3
from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT / "cmake" / "Versions.cmake"
PYPROJECT = ROOT / "pyproject.toml"


def cmake_var(text: str, name: str) -> str:
    match = re.search(
        rf'^\s*set\s*\(\s*{re.escape(name)}\s+"?([^")\s]+)"?\s*\)',
        text,
        re.MULTILINE,
    )
    if not match:
        raise SystemExit(f"missing {name} in {VERSIONS}")
    return match.group(1)


def expected_header(version: str) -> int:
    parts = version.split(".")
    if len(parts) != 3:
        raise SystemExit(f"invalid MuJoCo version: {version}")
    major, minor, patch = (int(part) for part in parts)
    return major * 1_000_000 + minor * 1_000 + patch


def main() -> int:
    versions_text = VERSIONS.read_text(encoding="utf-8")
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    project_version = cmake_var(versions_text, "MJ_KDL_VERSION")
    mujoco_version = cmake_var(versions_text, "MJ_KDL_MUJOCO_VERSION")

    py_project = pyproject["project"]
    errors: list[str] = []
    if py_project["version"] != project_version:
        errors.append(
            f"pyproject project.version={py_project['version']} != MJ_KDL_VERSION={project_version}"
        )

    dependency = f"mujoco=={mujoco_version}"
    if dependency not in py_project.get("dependencies", []):
        errors.append(f"pyproject dependencies must contain {dependency!r}")

    header_match = re.search(r"MJ_KDL_MUJOCO_VERSION_HEADER", versions_text)
    if not header_match:
        errors.append("Versions.cmake must compute MJ_KDL_MUJOCO_VERSION_HEADER")

    # Keep this calculation mirrored here so accidental formula edits are caught
    # by tests that inspect generated CMake cache values.
    expected_header(mujoco_version)

    if errors:
        raise SystemExit("\n".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
