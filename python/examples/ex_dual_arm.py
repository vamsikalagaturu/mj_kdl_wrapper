#!/usr/bin/env python3
"""Dual-arm example ported from src/examples/ex_dual_arm.cpp.

Two Kinova GEN3 arms, each with a Robotiq 2F-85 gripper, in one scene. Both run
joint-space impedance to the home pose. An Env on_reset hook re-homes both arms.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import mj_kdl_wrapper as mjk

DEFAULT_MODEL = "third_party/menagerie/kinova_gen3/gen3.xml"
DEFAULT_GRIPPER = "assets/robotiq_2f85/2f85.xml"
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


def attachment_gripper(path: Path, prefix: str = "g_") -> mjk.AttachmentSpec:
    spec = mjk.AttachmentSpec()
    spec.mjcf_path = str(path)
    spec.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
    spec.prefix = prefix
    return spec


def apply(robot: mjk.Robot, target: list[float]) -> None:
    robot.update()
    grav = robot.gravity_torques(-9.81)
    robot.jnt_trq_cmd = [
        KP[i] * (target[i] - robot.jnt_pos_msr[i]) - KD[i] * robot.jnt_vel_msr[i] + grav[i]
        for i in range(robot.n_joints)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    arm_path = path_from_arg(mjk.menagerie.model_path("kinova_gen3", env_var=MODEL_ENV_VAR), "arm model")
    gripper_path = path_from_arg(os.environ.get(GRIPPER_ENV_VAR, DEFAULT_GRIPPER), "gripper model")
    attach = attachment_gripper(gripper_path)

    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    left = mjk.RobotSpec()
    left.path = str(arm_path)
    left.pos = [-1.0, 0.0, 0.0]
    left.attachments = [attach]
    right = mjk.RobotSpec()
    right.path = str(arm_path)
    right.prefix = "r2_"
    right.pos = [1.0, 0.0, 0.0]
    right.attachments = [attach]
    spec.robots = [left, right]

    env = mjk.Env.build(spec)
    try:
        tool1 = mjk.ToolFrameSpec()
        tool1.tool_body = "g_base"
        tool1.tcp_site = "g_pinch"
        tool2 = mjk.ToolFrameSpec()
        tool2.tool_body = "r2_g_base"
        tool2.tcp_site = "r2_g_pinch"
        arm1 = env.create_robot("base_link", "bracelet_link", tool=tool1)
        arm2 = env.create_robot("r2_base_link", "r2_bracelet_link", tool=tool2)
        for robot in (arm1, arm2):
            robot.ctrl_mode = mjk.CtrlMode.TORQUE

        def on_reset(ctx):
            arm1.set_joint_pos(HOME_POSE, call_forward=False)
            arm2.set_joint_pos(HOME_POSE, call_forward=False)

        env.on_reset = on_reset
        env.reset()

        def step():
            apply(arm1, HOME_POSE)
            apply(arm2, HOME_POSE)
            grip = 255.0 if math.fmod(env.time(), 6.0) < 3.0 else 0.0
            for name in ("g_fingers_actuator", "r2_g_fingers_actuator"):
                if env.has_actuator(name):
                    env.set_actuator_ctrl(name, grip)

        if args.gui:
            viewer = mjk.SimulateViewer.open(arm1, "ex_dual_arm.py")
            prev = env.time()
            try:
                while viewer.is_running():
                    if env.time() < prev - 1e-6:
                        env.reset()
                    prev = env.time()
                    step()
                    if not viewer.step():
                        break
            finally:
                viewer.close()
        else:
            end = env.time() + 1.2
            while env.time() < end:
                step()
                arm1.step()
            arm1_frame = arm1.fk_frame()
            arm2_frame = arm2.fk_frame()
            arm1_pos = [arm1_frame.p.x(), arm1_frame.p.y(), arm1_frame.p.z()]
            arm2_pos = [arm2_frame.p.x(), arm2_frame.p.y(), arm2_frame.p.z()]
            print(f"arm1 EE: {[round(x, 4) for x in arm1_pos]}")
            print(f"arm2 EE: {[round(x, 4) for x in arm2_pos]}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
