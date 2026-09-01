#!/usr/bin/env python3
"""Launch the mj_kdl_wrapper custom Simulate UI from Python."""

from __future__ import annotations

import mj_kdl_wrapper as mjk

TITLE = "mj_kdl_wrapper Python UI"


def main() -> int:
    model_path = mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")

    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    robot_spec = mjk.RobotSpec()
    robot_spec.path = model_path
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
                viewer.pace()
        finally:
            viewer.close()
    finally:
        scene.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
