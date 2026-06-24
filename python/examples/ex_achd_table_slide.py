#!/usr/bin/env python3
"""ACHD table-slide example using PyKDL's Vereshchagin solver directly.

Ported from src/examples/ex_achd_table_slide.cpp. Slides the TCP along +X under
an acceleration-constrained hybrid-dynamics controller. An Env on_reset hook
restores the start pose (and re-primes the PID) so the simulate-UI reset replays
the slide.
"""

from __future__ import annotations

import argparse

import PyKDL as kdl
import mj_kdl_wrapper as mjk

TABLE_POSE = [-0.00258, 1.43, 3.14, -1.70, -0.018, 1.74, 1.57]
TABLE_Z = 0.447
MOVE_X = 0.20
KPLIN, KDLIN = 200.0, 30.0
KPROT, KDROT = 175.0, 28.0
BETA_MAX, TAU_MAX = 120.0, 59.0


def build_env() -> tuple[mjk.Env, mjk.Robot]:
    table = mjk.SceneObject()
    table.name = "table"
    table.mjcf_path = mjk.menagerie.asset_path("table.xml", env_var="MJ_KDL_TABLE")
    table.pos = [0.0, 0.0, TABLE_Z]
    table.fixed = True
    attach = mjk.AttachmentSpec()
    attach.mjcf_path = mjk.menagerie.asset_path("robotiq_2f85/2f85.xml", env_var="MJ_KDL_GRIPPER")
    attach.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
    attach.prefix = "g_"
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    spec.objects = [table]
    robot_spec = mjk.RobotSpec()
    robot_spec.path = mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")
    robot_spec.pos = [0.0, 0.0, TABLE_Z]
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


def alpha_no_linear_z() -> kdl.Jacobian:
    alpha = kdl.Jacobian(5)
    alpha.setColumn(0, kdl.Twist(kdl.Vector(1, 0, 0), kdl.Vector.Zero()))
    alpha.setColumn(1, kdl.Twist(kdl.Vector(0, 1, 0), kdl.Vector.Zero()))
    alpha.setColumn(2, kdl.Twist(kdl.Vector.Zero(), kdl.Vector(1, 0, 0)))
    alpha.setColumn(3, kdl.Twist(kdl.Vector.Zero(), kdl.Vector(0, 1, 0)))
    alpha.setColumn(4, kdl.Twist(kdl.Vector.Zero(), kdl.Vector(0, 0, 1)))
    return alpha


def achd_step(robot, chain, fk, achd, rnea, target, err_prev, first_pid):
    robot.update()
    q = jnt(robot.jnt_pos_msr)
    qd = jnt(robot.jnt_vel_msr)
    current = kdl.Frame()
    fk.JntToCart(q, current)
    err = kdl.diff(current, target)
    dt = 0.002
    e = [err.vel.x(), err.vel.y(), err.rot.x(), err.rot.y(), err.rot.z()]
    if first_pid[0]:
        err_prev[:] = e
        first_pid[0] = False
    de = [(e[i] - err_prev[i]) / dt for i in range(5)]
    err_prev[:] = e
    beta = kdl.JntArray(5)
    beta[0] = clamp_abs(KPLIN * e[0] + KDLIN * de[0], BETA_MAX)
    beta[1] = clamp_abs(KPLIN * e[1] + KDLIN * de[1], BETA_MAX)
    beta[2] = clamp_abs(KPROT * e[2] + KDROT * de[2], BETA_MAX)
    beta[3] = clamp_abs(KPROT * e[3] + KDROT * de[3], BETA_MAX)
    beta[4] = clamp_abs(KPROT * e[4] + KDROT * de[4], BETA_MAX)

    n = robot.n_joints
    qdd = kdl.JntArray(n)
    ff = kdl.JntArray(n)
    constraint_tau = kdl.JntArray(n)
    f_ext = [kdl.Wrench.Zero() for _ in range(chain.getNrOfSegments())]
    if achd.CartToJnt(q, qd, qdd, alpha_no_linear_z(), beta, f_ext, ff, constraint_tau) < 0:
        raise RuntimeError("PyKDL ACHD failed")

    tau = kdl.JntArray(n)
    rnea_wrenches = [kdl.Wrench.Zero() for _ in range(chain.getNrOfSegments())]
    if rnea.CartToJnt(q, qd, qdd, rnea_wrenches, tau) < 0:
        raise RuntimeError("PyKDL RNEA failed")
    robot.jnt_trq_cmd = [clamp_abs(tau[i], TAU_MAX) for i in range(n)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    env, robot = build_env()
    try:
        chain = robot.kdl_chain()
        fk = kdl.ChainFkSolverPos_recursive(chain)
        achd = kdl.ChainHdSolver_Vereshchagin_Fixed_Joint(
            chain, kdl.Twist(kdl.Vector(0.0, 0.0, 9.81), kdl.Vector.Zero()), 5
        )
        rnea = kdl.ChainIdSolver_RNE(chain, kdl.Vector(0.0, 0.0, -9.81))
        robot.ctrl_mode = mjk.CtrlMode.TORQUE

        err_prev = [0.0] * 5
        first_pid = [True]

        def on_reset(ctx):
            robot.set_joint_pos(TABLE_POSE, call_forward=False)
            first_pid[0] = True  # re-prime PID after a reset

        env.on_reset = on_reset
        env.reset()

        robot.update()
        start = kdl.Frame()
        fk.JntToCart(jnt(TABLE_POSE), start)
        target = kdl.Frame(start.M, start.p + kdl.Vector(MOVE_X, 0.0, 0.0))

        def step():
            achd_step(robot, chain, fk, achd, rnea, target, err_prev, first_pid)
            if env.has_actuator("g_fingers_actuator"):
                env.set_actuator_ctrl("g_fingers_actuator", 255.0)

        if args.gui:
            viewer = mjk.SimulateViewer.open(robot, "ex_achd_table_slide.py")
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
            end = env.time() + 2.0
            while env.time() < end:
                step()
                if not robot.step():
                    break
        print(f"tcp target x shift: {MOVE_X:.3f} m")
        final_frame = robot.fk_frame()
        final_pos = [final_frame.p.x(), final_frame.p.y(), final_frame.p.z()]
        print(f"final bracelet frame: {[round(x, 3) for x in final_pos]}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
