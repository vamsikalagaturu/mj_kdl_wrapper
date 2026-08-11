#!/usr/bin/env python3
"""Joint velocity-control example ported from src/examples/ex_vel_ctrl.cpp.

gen3.xml has POSITION actuators, so velocity control integrates a clamped
velocity command into the position setpoint each step. An Env on_reset hook
re-homes the arm and restarts the motion so the simulate-UI reset replays it.
"""

from __future__ import annotations

import argparse

import mj_kdl_wrapper as mjk

HOME_POSE = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
TARGET_POSE = [0.3, 0.5, 2.9, -2.0, 0.3, 1.2, 1.3]
KV = 2.0
MAX_VEL = 0.6
TOL = 0.01


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_env(model_path: str) -> tuple[mjk.Env, mjk.Robot]:
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    robot_spec = mjk.RobotSpec()
    robot_spec.path = model_path
    spec.robots = [robot_spec]
    env = mjk.Env.build(spec)
    robot = env.create_robot("base_link", "bracelet_link")
    robot.ctrl_mode = mjk.CtrlMode.POSITION
    return env, robot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Open the custom Simulate UI.")
    args = parser.parse_args()

    model_path = mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")
    env, robot = build_env(model_path)
    try:
        dt = env.timestep()
        state = {"arrived": False}

        def on_reset(ctx):
            robot.set_joint_pos(HOME_POSE, call_forward=False)
            robot.jnt_pos_cmd = HOME_POSE[:]
            state["arrived"] = False

        env.on_reset = on_reset
        env.reset()

        def control_step() -> None:
            robot.update()
            if state["arrived"]:
                return
            pos_cmd = robot.jnt_pos_cmd
            max_err = 0.0
            for i in range(robot.n_joints):
                err = TARGET_POSE[i] - robot.jnt_pos_msr[i]
                max_err = max(max_err, abs(err))
                pos_cmd[i] += clamp(KV * err, -MAX_VEL, MAX_VEL) * dt
            if max_err < TOL:
                robot.jnt_pos_cmd = robot.jnt_pos_msr
                state["arrived"] = True
            else:
                robot.jnt_pos_cmd = pos_cmd

        if args.gui:
            viewer = mjk.SimulateViewer.open(robot, "ex_vel_ctrl.py")
            prev = env.time()
            try:
                while viewer.is_running():
                    if env.time() < prev - 1e-6:
                        env.reset()
                    prev = env.time()
                    control_step()
                    if not viewer.step():
                        break
                    viewer.pace()
            finally:
                viewer.close()
        else:
            end = env.time() + 5.0
            while env.time() < end and not state["arrived"]:
                control_step()
                robot.step()
                robot.pace()
            max_err = max(abs(TARGET_POSE[i] - robot.jnt_pos_msr[i]) for i in range(robot.n_joints))
            status = "converged" if state["arrived"] else "timeout"
            print(f"max joint error: {max_err:.4f} rad  ({status})")
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
