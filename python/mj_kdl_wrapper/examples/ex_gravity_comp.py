#!/usr/bin/env python3
"""Gravity-compensation example ported from src/examples/ex_gravity_comp.cpp.

Uses an Env with an on_reset hook (re-homes the arm) so the simulate-UI reset
button restores the home pose, mirroring the C++ example.
"""

from __future__ import annotations

import argparse

import PyKDL as kdl

import mj_kdl_wrapper as mjk

HOME_POSE = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]


def joints(values) -> kdl.JntArray:
    q = kdl.JntArray(len(values))
    for i, value in enumerate(values):
        q[i] = value
    return q


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
    return env, robot


def run_loop(env: mjk.Env, step_fn, *, duration: float, gui: bool) -> None:
    if gui:
        # The UI's reset button runs env's reset, on_reset included.
        env.open_viewer("ex_gravity_comp.py")
    end = env.data.time + duration
    while env.data.time < end:
        step_fn()
        if not env.step():
            break
        env.pace()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    model_path = mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")
    env, robot = build_env(model_path)
    try:
        robot.set_control_mode(mjk.CtrlMode.TORQUE)
        env.on_reset = lambda ctx: robot.set_joint_pos(HOME_POSE)
        env.reset()

        chain = robot.kdl_chain()
        fk = kdl.ChainFkSolverPos_recursive(chain)
        dyn = kdl.ChainDynParam(chain, kdl.Vector(0.0, 0.0, -9.81))

        def tcp() -> kdl.Frame:
            frame = kdl.Frame()
            fk.JntToCart(joints(robot.jnt_pos_msr), frame)
            return frame

        env.update()
        start_frame = tcp()
        start = [start_frame.p.x(), start_frame.p.y(), start_frame.p.z()]

        def step():
            env.update()
            g = kdl.JntArray(robot.n_joints)
            dyn.JntToGravity(joints(robot.jnt_pos_msr), g)
            robot.jnt_trq_cmd = [g[i] for i in range(robot.n_joints)]
            env.update()

        run_loop(env, step, duration=2.0, gui=args.gui)
        end_frame = tcp()
        end = [end_frame.p.x(), end_frame.p.y(), end_frame.p.z()]
        drift = sum((end[i] - start[i]) ** 2 for i in range(3)) ** 0.5
        print(f"EE drift: {drift:.6f} m")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
