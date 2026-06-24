#!/usr/bin/env python3
"""Joint-space impedance example ported from src/examples/ex_impedance.cpp.

tau = Kp*(q_home - q) - Kd*qdot + g_kdl, applied in TORQUE mode, with the
gripper cycling open/closed. An Env on_reset hook re-homes the arm.
"""

from __future__ import annotations

import argparse
import math

import mj_kdl_wrapper as mjk

HOME_POSE = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]

KP = [100, 200, 100, 200, 100, 200, 100]
KD = [10, 20, 10, 20, 10, 20, 10]


def attachment_gripper(path: str) -> mjk.AttachmentSpec:
    spec = mjk.AttachmentSpec()
    spec.mjcf_path = path
    spec.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
    spec.prefix = "g_"
    return spec


def build_env(model_path: str, gripper_path: str) -> tuple[mjk.Env, mjk.Robot]:
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    robot_spec = mjk.RobotSpec()
    robot_spec.path = model_path
    robot_spec.attachments = [attachment_gripper(gripper_path)]
    spec.robots = [robot_spec]
    env = mjk.Env.build(spec)
    tool = mjk.ToolFrameSpec()
    tool.tool_body = "g_base"
    tool.tcp_site = "g_pinch"
    robot = env.create_robot("base_link", "bracelet_link", tool=tool)
    return env, robot


def apply_pd_gravity(robot: mjk.Robot, target: list[float]) -> None:
    robot.update()
    grav = robot.gravity_torques(-9.81)
    robot.jnt_trq_cmd = [
        KP[i] * (target[i] - robot.jnt_pos_msr[i]) - KD[i] * robot.jnt_vel_msr[i] + grav[i]
        for i in range(robot.n_joints)
    ]


def run_loop(env: mjk.Env, robot: mjk.Robot, step_fn, *, duration: float, gui: bool) -> None:
    if gui:
        viewer = mjk.SimulateViewer.open(robot, "ex_impedance.py")
        prev = env.time()
        try:
            while viewer.is_running():
                if env.time() < prev - 1e-6:
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

    env, robot = build_env(
        mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL"),
        mjk.menagerie.asset_path("robotiq_2f85/2f85.xml", env_var="MJ_KDL_GRIPPER"),
    )
    try:
        robot.ctrl_mode = mjk.CtrlMode.TORQUE
        env.on_reset = lambda ctx: robot.set_joint_pos(HOME_POSE, call_forward=False)
        env.reset()

        def step():
            apply_pd_gravity(robot, HOME_POSE)
            if env.has_actuator("g_fingers_actuator"):
                env.set_actuator_ctrl(
                    "g_fingers_actuator", 255.0 if math.fmod(env.time(), 6.0) < 3.0 else 0.0
                )

        run_loop(env, robot, step, duration=3.0, gui=args.gui)
        print(f"final q: {[round(x, 4) for x in robot.jnt_pos_msr]}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
