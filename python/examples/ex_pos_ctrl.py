#!/usr/bin/env python3
"""Joint position-control example ported from src/examples/ex_pos_ctrl.cpp."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mj_kdl_wrapper as mjk

DEFAULT_MODEL = "third_party/menagerie/kinova_gen3/gen3.xml"
MODEL_ENV_VAR = "MJ_KDL_MODEL"

HOME_POSE = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
TARGET_POSE = [0.3, 0.5, 2.9, -2.0, 0.3, 1.2, 1.3]
MOTION_DURATION = 2.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_robot(model_path: Path) -> tuple[mjk.Scene, mjk.Robot]:
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True

    robot_spec = mjk.RobotSpec()
    robot_spec.path = str(model_path)
    spec.robots = [robot_spec]

    scene = mjk.Scene.build(spec)
    robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link")
    robot.ctrl_mode = mjk.CtrlMode.POSITION
    robot.set_joint_pos(HOME_POSE, call_forward=False)
    robot.jnt_pos_cmd = HOME_POSE[:]
    return scene, robot


def control_step(robot: mjk.Robot, sim_time: float) -> None:
    robot.update()
    alpha = clamp(sim_time / MOTION_DURATION, 0.0, 1.0)
    robot.jnt_pos_cmd = [
        HOME_POSE[i] + alpha * (TARGET_POSE[i] - HOME_POSE[i])
        for i in range(robot.n_joints)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Open the custom Simulate UI.")
    args = parser.parse_args()

    model_path = Path(os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL))
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} does not exist. Run from a directory where that relative path exists "
            f"or set {MODEL_ENV_VAR}."
        )

    scene, robot = build_robot(model_path)
    try:
        dt = scene.spec.timestep
        sim_time = 0.0

        if args.gui:
            viewer = mjk.SimulateViewer.open(robot, "ex_pos_ctrl.py")
            try:
                while viewer.is_running():
                    control_step(robot, sim_time)
                    if not viewer.step():
                        break
                    sim_time += dt
            finally:
                viewer.close()
        else:
            while sim_time < MOTION_DURATION + 1.0:
                control_step(robot, sim_time)
                robot.step()
                sim_time += dt

            max_err = max(abs(TARGET_POSE[i] - robot.jnt_pos_msr[i]) for i in range(robot.n_joints))
            print(f"max joint error at end: {max_err:.4f} rad")
    finally:
        scene.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
