#!/usr/bin/env python3
"""ACHD table-slide example using PyKDL's Vereshchagin solver directly.

Ported from src/examples/ex_achd_table_slide.cpp. The arm settles on the table under its
position servos, then slides its TCP 0.2 m along +X in TORQUE mode under ACHD (linear Z left
free) + RNEA while pressing down with PRESS_FORCE through ACHD's external-force input (driver
weights 1, the pinned KDL fork's setDriverWeights). Over the second half of the slide it
measures the table's contact normal force with mujoco.mj_contactForce on env.model/env.data,
prints the mean against the command, and exits 1 if the contact was not held.
"""

from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np
import PyKDL as kdl
import mj_kdl_wrapper as mjk

TABLE_POSE = [-0.00258, 1.43, 3.14, -1.70, -0.018, 1.74, 1.57]
TABLE_Z = 0.447
MOVE_X = 0.20
V_MAX_LIN = 0.08  # [m/s] the tracked reference moves toward the target at this speed
KPLIN, KDLIN = 200.0, 30.0
KPROT, KDROT = 175.0, 28.0
BETA_MAX = 120.0
PRESS_FORCE = 10.0  # [N] commanded straight down at the TCP
SETTLE_STEPS = 300
SLIDE_STEPS = 2000
CONTACT_HELD = 0.5  # the contact chatters while sliding


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
    tool.tool_body = "g_base_mount"
    tool.tcp_site = "g_pinch"
    robot = env.create_robot("base_link", "bracelet_link", tool=tool)
    return env, robot


def jnt(values) -> kdl.JntArray:
    out = kdl.JntArray(len(values))
    for i, v in enumerate(values):
        out[i] = v
    return out


def clamp_abs(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


# Sum of contact normal forces on one geom; positive pushes the bodies apart.
def contact_normal_force(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> float:
    total = 0.0
    force = np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        if geom not in (contact.geom1, contact.geom2):
            continue
        mujoco.mj_contactForce(model, data, i, force)
        total += force[0]
    return total


def alpha_no_linear_z() -> kdl.Jacobian:
    alpha = kdl.Jacobian(5)
    alpha.setColumn(0, kdl.Twist(kdl.Vector(1, 0, 0), kdl.Vector.Zero()))
    alpha.setColumn(1, kdl.Twist(kdl.Vector(0, 1, 0), kdl.Vector.Zero()))
    alpha.setColumn(2, kdl.Twist(kdl.Vector.Zero(), kdl.Vector(1, 0, 0)))
    alpha.setColumn(3, kdl.Twist(kdl.Vector.Zero(), kdl.Vector(0, 1, 0)))
    alpha.setColumn(4, kdl.Twist(kdl.Vector.Zero(), kdl.Vector(0, 0, 1)))
    return alpha


class Slide:
    """The ACHD controller and the press measurement; restart() replays it from the current pose."""

    def __init__(self, env: mjk.Env, robot: mjk.Robot):
        self.env, self.robot = env, robot
        self.chain = robot.kdl_chain()
        self.fk = kdl.ChainFkSolverPos_recursive(self.chain)
        gravity_z = env.model.opt.gravity[2]
        self.achd = kdl.ChainHdSolver_Vereshchagin(
            self.chain, kdl.Twist(kdl.Vector(0.0, 0.0, -gravity_z), kdl.Vector.Zero()), 5
        )
        # Driver weights 1 pass the commanded wrench through to the environment.
        self.achd.setDriverWeights(np.ones(5), np.zeros(5))
        self.rnea = kdl.ChainIdSolver_RNE(self.chain, kdl.Vector(0.0, 0.0, gravity_z))
        self.alpha = alpha_no_linear_z()
        segments = self.chain.getNrOfSegments()
        self.f_ext = [kdl.Wrench.Zero() for _ in range(segments)]
        self.f_ext[-1] = kdl.Wrench(kdl.Vector(0.0, 0.0, -PRESS_FORCE), kdl.Vector.Zero())
        self.no_wrench = [kdl.Wrench.Zero() for _ in range(segments)]
        self.table_geom = env.model.geom("top").id
        self.restart()
        self.target = kdl.Frame(self.tracked.M, self.tracked.p + kdl.Vector(MOVE_X, 0.0, 0.0))

    def tcp(self) -> kdl.Frame:
        frame = kdl.Frame()
        self.fk.JntToCart(jnt(self.robot.jnt_pos_msr), frame)
        return frame

    def restart(self) -> None:
        self.env.update()
        self.tracked = self.tcp()
        self.err_prev = None
        self.steps = 0
        self.contact_steps = 0
        self.reaction_sum = 0.0
        self.reaction_count = 0

    def control(self) -> None:
        env, robot = self.env, self.robot
        dt = env.model.opt.timestep
        to_goal = self.target.p - self.tracked.p
        dist = to_goal.Norm()
        if dist > 1e-4:
            self.tracked.p += to_goal * (min(dist, V_MAX_LIN * dt) / dist)

        env.update()
        q, qd = jnt(robot.jnt_pos_msr), jnt(robot.jnt_vel_msr)
        current = kdl.Frame()
        self.fk.JntToCart(q, current)
        err = kdl.diff(current, self.tracked)
        e = [err.vel.x(), err.vel.y(), err.rot.x(), err.rot.y(), err.rot.z()]
        prev = self.err_prev or e
        de = [(e[i] - prev[i]) / dt for i in range(5)]
        self.err_prev = e
        gains = [(KPLIN, KDLIN)] * 2 + [(KPROT, KDROT)] * 3
        beta = kdl.JntArray(5)
        for i, (kp, kd) in enumerate(gains):
            beta[i] = clamp_abs(kp * e[i] + kd * de[i], BETA_MAX)

        n = robot.n_joints
        # Gravity as feed-forward, so only the commanded press pushes the free linear Z down.
        ff = kdl.JntArray(n)
        if self.rnea.CartToJnt(q, kdl.JntArray(n), kdl.JntArray(n), self.no_wrench, ff) < 0:
            raise RuntimeError("PyKDL RNEA failed")
        qdd, constraint_tau, tau = kdl.JntArray(n), kdl.JntArray(n), kdl.JntArray(n)
        if self.achd.CartToJnt(q, qd, qdd, self.alpha, beta, self.f_ext, ff, constraint_tau) < 0:
            raise RuntimeError("PyKDL ACHD failed")
        if self.rnea.CartToJnt(q, qd, qdd, self.no_wrench, tau) < 0:
            raise RuntimeError("PyKDL RNEA failed")
        # update() clamps each torque to its joint's limit and reports it in jnt_saturated.
        robot.jnt_trq_cmd = [tau[i] for i in range(n)]
        env.update()
        self.steps += 1

    # Measured over the second half of the slide: the press first has to bring the TCP down.
    def measure(self) -> None:
        if self.steps <= SLIDE_STEPS // 2:
            return
        reaction = contact_normal_force(self.env.model, self.env.data, self.table_geom)
        self.contact_steps += reaction > 0.0
        self.reaction_sum += reaction
        self.reaction_count += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    env, robot = build_env()
    try:
        restarted = [False]

        def on_reset(ctx):
            robot.set_joint_pos(TABLE_POSE)
            restarted[0] = True

        env.on_reset = on_reset
        env.reset()
        # Let contacts settle with the table, the position servos holding the pose, before
        # starting the horizontal task in torque mode.
        for _ in range(SETTLE_STEPS):
            env.step()
        robot.set_control_mode(mjk.CtrlMode.TORQUE)
        slide = Slide(env, robot)

        if args.gui:
            # The UI's reset button runs env's reset, on_reset included.
            env.open_viewer("ex_achd_table_slide.py")
        restarted[0] = False
        while slide.steps < SLIDE_STEPS:
            slide.control()
            if not env.step():
                break
            slide.measure()
            if restarted[0]:
                restarted[0] = False
                slide.restart()
            env.pace()

        env.update()
        err = kdl.diff(slide.tcp(), slide.target)
        xy_err = math.hypot(err.vel.x(), err.vel.y())
        print(
            f"tcp_xy_err_mm={xy_err * 1000:.3f} tcp_z_error_unconstrained_mm="
            f"{err.vel.z() * 1000:.3f} tcp_rot_err_rad={err.rot.Norm():.3f}"
        )
        count = max(slide.reaction_count, 1)
        contact = slide.contact_steps / count
        mean = slide.reaction_sum / count
        print(
            f"table_contact_fraction={contact:.3f} mean_table_reaction_N={mean:.3f} "
            f"commanded_press_N={PRESS_FORCE:.3f} reaction_over_command={mean / PRESS_FORCE:.3f}"
        )
    finally:
        env.close()
    if contact < CONTACT_HELD:
        print("table contact not held during the slide")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
