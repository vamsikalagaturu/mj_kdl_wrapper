#!/usr/bin/env python3
"""Gravity-compensation example ported from src/examples/ex_gravity_comp.cpp."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mj_kdl_wrapper as mjk

DEFAULT_MODEL = "third_party/menagerie/kinova_gen3/gen3.xml"
MODEL_ENV_VAR = "MJ_KDL_MODEL"
HOME_POSE = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]


def path_from_arg(value: str, label: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{value} does not exist for {label}")
    return path


def build_scene(model_path: Path) -> tuple[mjk.Scene, mjk.Robot]:
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    robot_spec = mjk.RobotSpec()
    robot_spec.path = str(model_path)
    spec.robots = [robot_spec]
    scene = mjk.Scene.build(spec)
    robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link")
    return scene, robot


def run_loop(scene: mjk.Scene, robot: mjk.Robot, step_fn, *, duration: float, gui: bool) -> None:
    if gui:
        viewer = mjk.SimulateViewer.open(robot, "ex_gravity_comp.py")
        try:
            while viewer.is_running():
                step_fn()
                if not viewer.step():
                    break
        finally:
            viewer.close()
        return
    end = scene.time() + duration
    while scene.time() < end:
        step_fn()
        robot.step()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    model_path = path_from_arg(os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL), "arm model")
    scene, robot = build_scene(model_path)
    try:
        robot.ctrl_mode = mjk.CtrlMode.TORQUE
        robot.set_joint_pos(HOME_POSE, call_forward=False)
        robot.update()
        start_frame = robot.fk_frame()
        start = [start_frame.p.x(), start_frame.p.y(), start_frame.p.z()]

        def step():
            robot.update()
            robot.jnt_trq_cmd = robot.gravity_torques(-9.81)

        run_loop(scene, robot, step, duration=2.0, gui=args.gui)
        end_frame = robot.fk_frame()
        end = [end_frame.p.x(), end_frame.p.y(), end_frame.p.z()]
        drift = sum((end[i] - start[i]) ** 2 for i in range(3)) ** 0.5
        print(f"EE drift: {drift:.6f} m")
    finally:
        scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
