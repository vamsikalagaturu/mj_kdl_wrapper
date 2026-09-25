#!/usr/bin/env python3
"""Dual-arm example ported from src/examples/ex_dual_arm.cpp.

Two Kinova GEN3 arms, each with a Robotiq 2F-85 gripper, in one scene. Both run
joint-space impedance to the home pose. An Env on_reset hook re-homes both arms.
"""

from __future__ import annotations

import argparse
import math

import PyKDL as kdl

import mj_kdl_wrapper as mjk

HOME_POSE = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]

KP = [100, 200, 100, 200, 100, 200, 100]
KD = [10, 20, 10, 20, 10, 20, 10]
GRIPPER_CLOSED = 0.82  # [rad] driver joint; the bundled 2F-85's ctrlrange is 0..0.82


def attachment_gripper(path: str, prefix: str = "g_") -> mjk.AttachmentSpec:
    spec = mjk.AttachmentSpec()
    spec.mjcf_path = path
    spec.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
    spec.prefix = prefix
    return spec


def joints(values) -> kdl.JntArray:
    q = kdl.JntArray(len(values))
    for i, value in enumerate(values):
        q[i] = value
    return q


def impedance(robot: mjk.Robot, dyn: kdl.ChainDynParam, target: list[float]) -> None:
    grav = kdl.JntArray(robot.n_joints)
    dyn.JntToGravity(joints(robot.jnt_pos_msr), grav)
    robot.jnt_trq_cmd = [
        KP[i] * (target[i] - robot.jnt_pos_msr[i]) - KD[i] * robot.jnt_vel_msr[i] + grav[i]
        for i in range(robot.n_joints)
    ]


def tcp_position(robot: mjk.Robot, fk: kdl.ChainFkSolverPos_recursive) -> list[float]:
    frame = kdl.Frame()
    fk.JntToCart(joints(robot.jnt_pos_msr), frame)
    return [frame.p.x(), frame.p.y(), frame.p.z()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    arm_path = mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")
    gripper_path = mjk.menagerie.asset_path("robotiq_2f85/2f85.xml", env_var="MJ_KDL_GRIPPER")
    attach = attachment_gripper(gripper_path)

    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    left = mjk.RobotSpec()
    left.path = arm_path
    left.pos = [-1.0, 0.0, 0.0]
    left.attachments = [attach]
    right = mjk.RobotSpec()
    right.path = arm_path
    right.prefix = "r2_"
    right.pos = [1.0, 0.0, 0.0]
    right.attachments = [attach]
    spec.robots = [left, right]

    env = mjk.Env.build(spec)
    try:
        tool1 = mjk.ToolFrameSpec()
        tool1.tool_body = "g_base_mount"
        tool1.tcp_site = "g_pinch"
        tool2 = mjk.ToolFrameSpec()
        tool2.tool_body = "r2_g_base_mount"
        tool2.tcp_site = "r2_g_pinch"
        arm1 = env.create_robot("base_link", "bracelet_link", tool=tool1)
        arm2 = env.create_robot("r2_base_link", "r2_bracelet_link", tool=tool2)
        for robot in (arm1, arm2):
            robot.set_control_mode(mjk.CtrlMode.TORQUE)

        def on_reset(ctx):
            arm1.set_joint_pos(HOME_POSE)
            arm2.set_joint_pos(HOME_POSE)

        env.on_reset = on_reset
        env.reset()
        chains = [arm1.kdl_chain(), arm2.kdl_chain()]
        dyns = [kdl.ChainDynParam(c, kdl.Vector(0.0, 0.0, -9.81)) for c in chains]
        fks = [kdl.ChainFkSolverPos_recursive(c) for c in chains]

        def step():
            env.update()
            impedance(arm1, dyns[0], HOME_POSE)
            impedance(arm2, dyns[1], HOME_POSE)
            grip = GRIPPER_CLOSED if math.fmod(env.data.time, 6.0) < 3.0 else 0.0
            for name in ("g_fingers_actuator", "r2_g_fingers_actuator"):
                env.data.actuator(name).ctrl[0] = grip
            env.update()

        if args.gui:
            # The UI's reset button re-homes both arms through on_reset.
            env.open_viewer("ex_dual_arm.py")
        end = env.data.time + 1.2
        while env.data.time < end:
            step()
            if not env.step():
                break
            env.pace()
        arm1_pos = tcp_position(arm1, fks[0])
        arm2_pos = tcp_position(arm2, fks[1])
        print(f"arm1 EE: {[round(x, 4) for x in arm1_pos]}")
        print(f"arm2 EE: {[round(x, 4) for x in arm2_pos]}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
