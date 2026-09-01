#!/usr/bin/env python3
"""ACHD Cartesian pick-place example using PyKDL directly.

Ported from src/examples/ex_achd_pick_place.cpp. Uses the Vereshchagin
acceleration-constrained hybrid-dynamics solver plus RNEA for computed torques.
An Env on_reset hook re-homes the arm, re-poses the cube, and opens the gripper.
"""

from __future__ import annotations

import argparse

import PyKDL as kdl
import mj_kdl_wrapper as mjk

HOME = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
SURFACE_Z = 0.70
KP_LIN, KD_LIN = 160.0, 30.0
KP_ROT, KD_ROT = 100.0, 35.0
BETA_LIN_MAX, BETA_ROT_MAX, TAU_MAX = 100.0, 70.0, 59.0
CUBE_START = [0.40, 0.0, SURFACE_Z + 0.02]


class ResetRequested(Exception):
    """Raised when the simulate UI reset is detected, to restart the sequence."""


def build_env() -> tuple[mjk.Env, mjk.Robot]:
    table = mjk.SceneObject()
    table.name = "table"
    table.mjcf_path = mjk.menagerie.asset_path("table.xml", env_var="MJ_KDL_TABLE")
    table.pos = [0.0, 0.0, SURFACE_Z]
    table.fixed = True

    cube = mjk.SceneObject()
    cube.name = "cube"
    cube.shape = mjk.Shape.BOX
    cube.size = [0.02, 0.02, 0.02]
    cube.pos = CUBE_START[:]
    cube.rgba = [0.1, 0.35, 1.0, 1.0]
    cube.mass = 0.1
    cube.condim = mjk.Condim.Torsional
    cube.friction = [0.8, 0.02, 0.001]

    attach = mjk.AttachmentSpec()
    attach.mjcf_path = mjk.menagerie.asset_path("robotiq_2f85/2f85.xml", env_var="MJ_KDL_GRIPPER")
    attach.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
    attach.prefix = "g_"

    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    spec.objects = [table, cube]
    robot_spec = mjk.RobotSpec()
    robot_spec.path = mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")
    robot_spec.pos = [0.0, 0.0, SURFACE_Z]
    robot_spec.attachments = [attach]
    spec.robots = [robot_spec]

    env = mjk.Env.build(spec)
    tool = mjk.ToolFrameSpec()
    tool.tool_body = "g_base"
    tool.tcp_site = "g_pinch"
    robot = env.create_robot("base_link", "bracelet_link", tool=tool)
    return env, robot


def jnt(values: list[float]) -> kdl.JntArray:
    out = kdl.JntArray(len(values))
    for i, v in enumerate(values):
        out[i] = v
    return out


def clamp_abs(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def smoothstep(value: float) -> float:
    t = max(0.0, min(1.0, value))
    return t * t * (3.0 - 2.0 * t)


def alpha_identity() -> kdl.Jacobian:
    alpha = kdl.Jacobian(6)
    for i in range(6):
        alpha[i, i] = 1.0
    return alpha


def achd_step(robot, chain, fk, achd, rnea, alpha, target, err_prev, first_pid) -> None:
    robot.update()
    n = robot.n_joints
    q = jnt(robot.jnt_pos_msr)
    qd = jnt(robot.jnt_vel_msr)

    current = kdl.Frame()
    fk.JntToCart(q, current)
    err = kdl.diff(current, target)
    dt = 0.002
    e = [err.vel.x(), err.vel.y(), err.vel.z(), err.rot.x(), err.rot.y(), err.rot.z()]
    if first_pid[0]:
        err_prev[:] = e
        first_pid[0] = False
    de = [(e[i] - err_prev[i]) / dt for i in range(6)]
    err_prev[:] = e

    beta = kdl.JntArray(6)
    beta[0] = clamp_abs(KP_LIN * e[0] + KD_LIN * de[0], BETA_LIN_MAX)
    beta[1] = clamp_abs(KP_LIN * e[1] + KD_LIN * de[1], BETA_LIN_MAX)
    beta[2] = clamp_abs(KP_LIN * e[2] + KD_LIN * de[2], BETA_LIN_MAX)
    beta[3] = clamp_abs(KP_ROT * e[3] + KD_ROT * de[3], BETA_ROT_MAX)
    beta[4] = clamp_abs(KP_ROT * e[4] + KD_ROT * de[4], BETA_ROT_MAX)
    beta[5] = clamp_abs(KP_ROT * e[5] + KD_ROT * de[5], BETA_ROT_MAX)

    qdd = kdl.JntArray(n)
    ff = kdl.JntArray(n)
    constraint_tau = kdl.JntArray(n)
    f_ext = [kdl.Wrench.Zero() for _ in range(chain.getNrOfSegments())]
    if achd.CartToJnt(q, qd, qdd, alpha, beta, f_ext, ff, constraint_tau) < 0:
        raise RuntimeError("PyKDL ACHD failed")

    tau = kdl.JntArray(n)
    if rnea.CartToJnt(q, qd, qdd, f_ext, tau) < 0:
        raise RuntimeError("PyKDL RNEA failed")
    robot.jnt_trq_cmd = [clamp_abs(tau[i], TAU_MAX) for i in range(n)]


def step_once(env, robot, viewer, state) -> bool:
    ok = viewer.step() if viewer is not None else robot.step()
    if not ok:
        return False
    if viewer is not None and env.time() < state["prev"] - 1e-6:
        env.reset()
        state["prev"] = env.time()
        raise ResetRequested()
    state["prev"] = env.time()
    return True


def run_phase(env, robot, chain, fk, achd, rnea, alpha, phase, viewer, state) -> bool:
    print(f"State: {phase['name']}")
    start = robot.fk_frame()
    t0 = env.time()
    err_prev = [0.0] * 6
    first_pid = [True]
    while env.time() - t0 < phase["duration"]:
        t = smoothstep((env.time() - t0) / phase["duration"])
        target = kdl.addDelta(start, kdl.diff(start, phase["target"]), t)
        achd_step(robot, chain, fk, achd, rnea, alpha, target, err_prev, first_pid)
        if env.has_actuator("g_fingers_actuator"):
            env.set_actuator_ctrl("g_fingers_actuator", phase["gripper"])
        if viewer is not None and not viewer.is_running():
            return False
        if not step_once(env, robot, viewer, state):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    env, robot = build_env()
    try:
        chain = robot.kdl_chain()
        fk = kdl.ChainFkSolverPos_recursive(chain)
        achd = kdl.ChainHdSolver_Vereshchagin_Fixed_Joint(
            chain, kdl.Twist(kdl.Vector(0.0, 0.0, 9.81), kdl.Vector.Zero()), 6
        )
        rnea = kdl.ChainIdSolver_RNE(chain, kdl.Vector(0.0, 0.0, -9.81))
        alpha = alpha_identity()

        robot.ctrl_mode = mjk.CtrlMode.TORQUE

        def on_reset(ctx):
            robot.set_joint_pos(HOME, call_forward=False)
            env.set_body_pose("cube", CUBE_START)
            if env.has_actuator("g_fingers_actuator"):
                env.set_actuator_ctrl("g_fingers_actuator", 0.0)

        env.on_reset = on_reset
        env.reset()

        robot.update()
        grasp_rot = robot.fk_frame().M

        def target(x: float, y: float, z: float) -> kdl.Frame:
            return kdl.Frame(grasp_rot, kdl.Vector(x, y, z))

        phases = [
            {"name": "HOME", "target": robot.fk_frame(), "duration": 0.8, "gripper": 0.0},
            {"name": "PICK_ABOVE", "target": target(0.40, 0.0, 0.24), "duration": 2.0, "gripper": 0.0},
            {"name": "PICK", "target": target(0.40, 0.0, 0.04), "duration": 1.8, "gripper": 0.0},
            {"name": "CLOSE", "target": target(0.40, 0.0, 0.04), "duration": 1.0, "gripper": 255.0},
            {"name": "LIFT", "target": target(0.40, 0.0, 0.34), "duration": 1.6, "gripper": 255.0},
            {"name": "PLACE_ABOVE", "target": target(0.40, 0.24, 0.24), "duration": 1.8, "gripper": 255.0},
            {"name": "PLACE", "target": target(0.40, 0.24, 0.04), "duration": 1.8, "gripper": 255.0},
            {"name": "OPEN", "target": target(0.40, 0.24, 0.04), "duration": 0.8, "gripper": 0.0},
            {"name": "RETREAT", "target": target(0.40, 0.24, 0.24), "duration": 1.2, "gripper": 0.0},
        ]

        state = {"prev": env.time()}
        if args.gui:
            viewer = mjk.SimulateViewer.open(robot, "ex_achd_pick_place.py")
            try:
                while viewer.is_running():
                    try:
                        for phase in phases:
                            if not run_phase(env, robot, chain, fk, achd, rnea, alpha, phase, viewer, state):
                                raise StopIteration
                        break
                    except ResetRequested:
                        continue
                    except StopIteration:
                        break
            finally:
                viewer.close()
        else:
            for phase in phases:
                if not run_phase(env, robot, chain, fk, achd, rnea, alpha, phase, None, state):
                    break
        cube = env.body_frame("cube")
        print(f"cube final position: {[round(v, 3) for v in [cube.p.x(), cube.p.y(), cube.p.z()]]}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
