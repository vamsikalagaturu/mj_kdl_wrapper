#!/usr/bin/env python3
"""Launch the mj_kdl_wrapper custom Simulate UI from Python."""

from __future__ import annotations

import os
from pathlib import Path

import mj_kdl_wrapper as mjk

DEFAULT_MODEL = "third_party/menagerie/kinova_gen3/gen3.xml"
MODEL_ENV_VAR = "MJ_KDL_MODEL"
TITLE = "mj_kdl_wrapper Python UI"


def main() -> int:
    model_path = Path(os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL))
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} does not exist. Run from a directory where that relative path exists "
            f"or set {MODEL_ENV_VAR}."
        )

    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    robot_spec = mjk.RobotSpec()
    robot_spec.path = str(model_path)
    spec.robots = [robot_spec]

    scene = mjk.Scene.build(spec)
    try:
        robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link")
        robot.jnt_pos_cmd = [0.0] * robot.n_joints

        viewer = mjk.SimulateViewer.open(robot, TITLE)
        try:
            while viewer.is_running():
                robot.update()
                if not viewer.step():
                    break
        finally:
            viewer.close()
    finally:
        scene.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
