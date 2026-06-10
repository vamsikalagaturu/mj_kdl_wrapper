#!/usr/bin/env python3
"""Minimal headless mj_kdl_wrapper Python example."""

from __future__ import annotations

import os
from pathlib import Path

import mj_kdl_wrapper as mjk


DEFAULT_MODEL = "third_party/menagerie/kinova_gen3/gen3.xml"
MODEL_ENV_VAR = "MJ_KDL_MODEL"


def resolve_model_path(path: str) -> Path:
    model_path = Path(path)
    if model_path.exists():
        return model_path
    return model_path


def main() -> int:
    model_path = resolve_model_path(mjk.menagerie.model_path("kinova_gen3", env_var=MODEL_ENV_VAR))
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

        for _ in range(10):
            robot.update()
            robot.step()

        print(f"joints: {robot.n_joints}")
        print(f"joint_names: {robot.joint_names}")
        print(f"q: {[round(x, 6) for x in robot.jnt_pos_msr]}")
        print(f"cameras: {scene.camera_names()}")
    finally:
        scene.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
