#!/usr/bin/env python3
"""Gravity-compensation example ported from src/examples/ex_gravity_comp.cpp.

Uses an Env with an on_reset hook (re-homes the arm) so the simulate-UI reset
button restores the home pose, mirroring the C++ example.
"""

from __future__ import annotations

import argparse

import mj_kdl_wrapper as mjk

HOME_POSE = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]


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


def run_loop(env: mjk.Env, robot: mjk.Robot, step_fn, *, duration: float, gui: bool) -> None:
    if gui:
        viewer = mjk.SimulateViewer.open(robot, "ex_gravity_comp.py")
        prev = env.time()
        try:
            while viewer.is_running():
                if env.time() < prev - 1e-6:  # user pressed reset in the UI
                    env.reset()
                prev = env.time()
                step_fn()
                if not viewer.step():
                    break
        finally:
            viewer.close()
        return
    end = env.time() + duration
    while env.time() < end:
        step_fn()
        robot.step()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    model_path = mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")
    env, robot = build_env(model_path)
    try:
        robot.ctrl_mode = mjk.CtrlMode.TORQUE
        env.on_reset = lambda ctx: robot.set_joint_pos(HOME_POSE, call_forward=False)
        env.reset()

        robot.update()
        start_frame = robot.fk_frame()
        start = [start_frame.p.x(), start_frame.p.y(), start_frame.p.z()]

        def step():
            robot.update()
            robot.jnt_trq_cmd = robot.gravity_torques(-9.81)

        run_loop(env, robot, step, duration=2.0, gui=args.gui)
        end_frame = robot.fk_frame()
        end = [end_frame.p.x(), end_frame.p.y(), end_frame.p.z()]
        drift = sum((end[i] - start[i]) ** 2 for i in range(3)) ** 0.5
        print(f"EE drift: {drift:.6f} m")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
