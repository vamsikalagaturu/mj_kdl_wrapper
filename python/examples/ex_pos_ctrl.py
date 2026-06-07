#!/usr/bin/env python3
"""Joint position-control example ported from src/examples/ex_pos_ctrl.cpp.

Drives the arm from home to a target with linearly interpolated position
setpoints. An Env on_reset hook re-homes the arm and restarts the motion clock
so the simulate-UI reset button replays the motion.
"""

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


def build_env(model_path: Path) -> tuple[mjk.Env, mjk.Robot]:
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    robot_spec = mjk.RobotSpec()
    robot_spec.path = str(model_path)
    spec.robots = [robot_spec]
    env = mjk.Env.build(spec)
    robot = env.create_robot("base_link", "bracelet_link")
    robot.ctrl_mode = mjk.CtrlMode.POSITION
    return env, robot


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

    env, robot = build_env(model_path)
    try:
        # t_start is reset on every env reset so the trajectory replays from home.
        state = {"t_start": 0.0}

        def on_reset(ctx):
            robot.set_joint_pos(HOME_POSE, call_forward=False)
            robot.jnt_pos_cmd = HOME_POSE[:]
            state["t_start"] = env.time()

        env.on_reset = on_reset
        env.reset()

        def control_step():
            robot.update()
            alpha = clamp((env.time() - state["t_start"]) / MOTION_DURATION, 0.0, 1.0)
            robot.jnt_pos_cmd = [
                HOME_POSE[i] + alpha * (TARGET_POSE[i] - HOME_POSE[i])
                for i in range(robot.n_joints)
            ]

        if args.gui:
            viewer = mjk.SimulateViewer.open(robot, "ex_pos_ctrl.py")
            prev = env.time()
            try:
                while viewer.is_running():
                    if env.time() < prev - 1e-6:
                        env.reset()
                    prev = env.time()
                    control_step()
                    if not viewer.step():
                        break
            finally:
                viewer.close()
        else:
            end = env.time() + MOTION_DURATION + 1.0
            while env.time() < end:
                control_step()
                robot.step()
            max_err = max(abs(TARGET_POSE[i] - robot.jnt_pos_msr[i]) for i in range(robot.n_joints))
            print(f"max joint error at end: {max_err:.4f} rad")
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
