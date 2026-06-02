#!/usr/bin/env python3
"""View a mj_kdl_wrapper-built scene with the official MuJoCo Python viewer."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import mj_kdl_wrapper as mjk

DEFAULT_MODEL = "third_party/menagerie/kinova_gen3/gen3.xml"
MODEL_ENV_VAR = "MJ_KDL_MODEL"


def resolve_model_path(path: str) -> Path:
    model_path = Path(path)
    if model_path.exists():
        return model_path
    return model_path


def build_scene(model_path: Path) -> mjk.Scene:
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    robot_spec = mjk.RobotSpec()
    robot_spec.path = str(model_path)
    spec.robots = [robot_spec]
    return mjk.Scene.build(spec)


def main() -> int:
    model_path = resolve_model_path(os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL))
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} does not exist. Run from a directory where that relative path exists "
            f"or set {MODEL_ENV_VAR}."
        )

    scene = build_scene(model_path)
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".mjb", delete=False)
        tmp.close()
        mjb_path = Path(tmp.name)
        scene.save_binary(str(mjb_path))
    finally:
        scene.close()

    try:
        viewer_code = """
import sys
import mujoco
import mujoco.viewer

expected = sys.argv[2]
actual = mujoco.mj_versionString()
if actual != expected:
    raise SystemExit(
        f"This example exports a MuJoCo {expected} .mjb file. "
        f"Installed Python mujoco is {actual}; install the same mujoco version "
        "or rebuild the wrapper against the installed Python mujoco version."
    )

model = mujoco.MjModel.from_binary_path(sys.argv[1])
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
"""
        try:
            subprocess.run(
                [sys.executable, "-c", viewer_code, str(mjb_path), mjk.mujoco_version()],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "MuJoCo viewer failed to start. If you are running from a sandbox, SSH session, "
                "or headless shell, run this command from a graphical session with DISPLAY or "
                "WAYLAND_DISPLAY available."
            ) from exc
    finally:
        if mjb_path.exists():
            mjb_path.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
