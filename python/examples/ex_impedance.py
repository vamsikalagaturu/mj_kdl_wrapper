#!/usr/bin/env python3
"""Joint-space impedance example ported from src/examples/ex_impedance.cpp."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import mj_kdl_wrapper as mjk

DEFAULT_MODEL = "third_party/menagerie/kinova_gen3/gen3.xml"
DEFAULT_GRIPPER = "third_party/menagerie/robotiq_2f85/2f85.xml"
MODEL_ENV_VAR = "MJ_KDL_MODEL"
GRIPPER_ENV_VAR = "MJ_KDL_GRIPPER"
HOME_POSE = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]

KP = [100, 200, 100, 200, 100, 200, 100]
KD = [10, 20, 10, 20, 10, 20, 10]


def path_from_arg(value: str, label: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{value} does not exist for {label}")
    return path


def attachment_gripper(path: Path) -> mjk.AttachmentSpec:
    spec = mjk.AttachmentSpec()
    spec.mjcf_path = str(path)
    spec.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
    spec.prefix = "g_"
    return spec


def build_scene(model_path: Path, gripper_path: Path) -> tuple[mjk.Scene, mjk.Robot]:
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    robot_spec = mjk.RobotSpec()
    robot_spec.path = str(model_path)
    robot_spec.attachments = [attachment_gripper(gripper_path)]
    spec.robots = [robot_spec]
    scene = mjk.Scene.build(spec)
    tool = mjk.ToolFrameSpec()
    tool.tool_body = "g_base"
    tool.tcp_site = "g_pinch"
    robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link", tool=tool)
    return scene, robot


def apply_pd_gravity(robot: mjk.Robot, target: list[float]) -> None:
    robot.update()
    grav = robot.gravity_torques(-9.81)
    robot.jnt_trq_cmd = [
        KP[i] * (target[i] - robot.jnt_pos_msr[i]) - KD[i] * robot.jnt_vel_msr[i] + grav[i]
        for i in range(robot.n_joints)
    ]


def run_loop(scene: mjk.Scene, robot: mjk.Robot, step_fn, *, duration: float, gui: bool) -> None:
    if gui:
        viewer = mjk.SimulateViewer.open(robot, "ex_impedance.py")
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

    scene, robot = build_scene(
        path_from_arg(os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL), "arm model"),
        path_from_arg(os.environ.get(GRIPPER_ENV_VAR, DEFAULT_GRIPPER), "gripper model"),
    )
    try:
        robot.ctrl_mode = mjk.CtrlMode.TORQUE
        robot.set_joint_pos(HOME_POSE, call_forward=False)

        def step():
            apply_pd_gravity(robot, HOME_POSE)
            if scene.has_actuator("g_fingers_actuator"):
                scene.set_actuator_ctrl(
                    "g_fingers_actuator", 255.0 if math.fmod(scene.time(), 6.0) < 3.0 else 0.0
                )

        run_loop(scene, robot, step, duration=3.0, gui=args.gui)
        print(f"final q: {[round(x, 4) for x in robot.jnt_pos_msr]}")
    finally:
        scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
