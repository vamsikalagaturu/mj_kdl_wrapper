#!/usr/bin/env python3
"""Joint POSITION then VELOCITY control on the Gen3; Python counterpart of ex_joint_ctrl.cpp."""

from __future__ import annotations

import argparse
import math

import mj_kdl_wrapper as mjk

HOME_POSE = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
TARGET_POSE = [0.3, 0.5, 2.9, -2.0, 0.3, 1.2, 1.3]
MOTION_DURATION = 2.0  # [s]
HOLD_TIME = 1.0  # [s]
MAX_ERR = 0.01  # [rad]
VEL_GAIN = 500.0  # velocity actuator kv [Nm s/rad]
KV = 2.0  # [rad/s per rad]
MAX_VEL = 0.6  # [rad/s]
TOL = 0.01  # [rad]
TIMEOUT = 5.0  # [s]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    robot_spec = mjk.RobotSpec()
    robot_spec.path = mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")
    velocity = mjk.CtrlModeSpec()
    velocity.mode = mjk.CtrlMode.VELOCITY
    velocity.kv = VEL_GAIN
    robot_spec.modes = [velocity]
    spec.robots = [robot_spec]

    env = mjk.Env.build(spec)
    try:
        robot = env.create_robot("base_link", "bracelet_link")
        state = {"restart": False}

        def on_reset(ctx):
            robot.set_joint_pos(HOME_POSE)
            robot.jnt_pos_cmd = HOME_POSE
            state["restart"] = True

        env.on_reset = on_reset
        env.reset()
        if args.gui:
            env.open_viewer("ex_joint_ctrl.py")

        motion, t_start = "position", 0.0
        pos_err = vel_err = math.inf
        vel_time = 0.0
        while True:
            if state["restart"]:
                state["restart"] = False
                robot.set_control_mode(mjk.CtrlMode.POSITION)
                motion, t_start = "position", env.data.time
                pos_err = vel_err = math.inf
            env.update()
            t = env.data.time - t_start
            if motion == "position":
                alpha = min(1.0, max(0.0, t / MOTION_DURATION))
                robot.jnt_pos_cmd = [h + alpha * (g - h) for h, g in zip(HOME_POSE, TARGET_POSE)]
                max_err = max(abs(g - q) for g, q in zip(TARGET_POSE, robot.jnt_pos_msr))
                if t >= MOTION_DURATION + HOLD_TIME:
                    pos_err = max_err
                    robot.set_control_mode(mjk.CtrlMode.VELOCITY)
                    motion, t_start = "velocity", env.data.time
                    continue
            else:
                errors = [h - q for h, q in zip(HOME_POSE, robot.jnt_pos_msr)]
                max_err = max(abs(e) for e in errors)
                robot.jnt_vel_cmd = [max(-MAX_VEL, min(MAX_VEL, KV * e)) for e in errors]
                if max_err < TOL or t >= TIMEOUT:
                    vel_err, vel_time = max_err, t
                    break
            if not env.step():
                break
            env.pace()
        pos_ok, vel_ok = pos_err <= MAX_ERR, vel_err < TOL
        status = "converged" if vel_ok else "not converged"
        print(f"POSITION: max joint error at the target {pos_err:.4f} rad (limit {MAX_ERR} rad)")
        print(
            f"VELOCITY: max joint error at home {vel_err:.4f} rad "
            f"({status} at t = {vel_time:.2f} s, timeout {TIMEOUT} s)"
        )
    finally:
        env.close()
    if args.gui:
        return 0
    ok = pos_ok and vel_ok
    print(f"{'PASS' if ok else 'FAIL'}: both motions reached their goal")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
