#!/usr/bin/env python3
"""Minimal headless mj_kdl_wrapper Python example."""

from __future__ import annotations

import mj_kdl_wrapper as mjk


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

        for _ in range(10):
            robot.update()
            robot.step()
            robot.pace()

        print(f"joints: {robot.n_joints}")
        print(f"joint_names: {robot.joint_names}")
        print(f"q: {[round(x, 6) for x in robot.jnt_pos_msr]}")
        print(f"cameras: {scene.camera_names()}")
    finally:
        scene.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
