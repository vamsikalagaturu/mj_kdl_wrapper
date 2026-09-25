#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT / "cmake" / "Versions.cmake"
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE_XML = ROOT / "package.xml"
WORKFLOWS = ROOT / ".github" / "workflows"
ROS2_WORKFLOW = WORKFLOWS / "ros2.yml"
MENAGERIE_PY = ROOT / "python" / "mj_kdl_wrapper" / "menagerie.py"


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

    # The header of the MuJoCo the build will use, when one is at hand (CI sets the variable).
    header = expected_header(mujoco_version)
    mujoco_h = Path(os.environ.get("MJ_KDL_MUJOCO_DIR", "/nonexistent")) / "include/mujoco/mujoco.h"
    if mujoco_h.exists():
        found = re.search(r"^#define mjVERSION_HEADER (\d+)", mujoco_h.read_text(), re.MULTILINE)
        if not found or int(found.group(1)) != header:
            errors.append(f"{mujoco_h}: mjVERSION_HEADER is not {header}")

    if not re.fullmatch(r"[0-9a-f]{64}", cmake_var(versions_text, "MJ_KDL_MUJOCO_SHA256")):
        errors.append("MJ_KDL_MUJOCO_SHA256 must be a sha256 hex digest")

    kdl_sha = cmake_var(versions_text, "MJ_KDL_OROCOS_KDL_GIT_SHA")
    if kdl_sha not in ROS2_WORKFLOW.read_text(encoding="utf-8"):
        errors.append(f"{ROS2_WORKFLOW.name} must check out MJ_KDL_OROCOS_KDL_GIT_SHA={kdl_sha}")

    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        for found in re.findall(r"MUJOCO_VERSION:\s*([0-9.]+)", workflow.read_text()):
            if found != mujoco_version:
                errors.append(f"{workflow.name}: MUJOCO_VERSION {found} != {mujoco_version}")

    package = ElementTree.parse(PACKAGE_XML).getroot().findtext("version")
    if package != project_version:
        errors.append(f"package.xml version={package} != MJ_KDL_VERSION={project_version}")

    menagerie_sha = cmake_var(versions_text, "MJ_KDL_MENAGERIE_GIT_SHA")
    if f'MENAGERIE_COMMIT = "{menagerie_sha}"' not in MENAGERIE_PY.read_text(encoding="utf-8"):
        errors.append(f"{MENAGERIE_PY.name}: MENAGERIE_COMMIT must be {menagerie_sha}")

    if errors:
        raise SystemExit("\n".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
