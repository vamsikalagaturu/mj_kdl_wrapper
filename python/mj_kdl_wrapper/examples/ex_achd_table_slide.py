#!/usr/bin/env python3
"""ACHD table slide pressing on the table; Python counterpart of ex_achd_table_slide.cpp.

The Gen3 has no gripper: the TCP is its wrist flange (the bracelet's pinch_site). A guarded
approach (all six directions tracked) lowers it onto the table, the press ramps up, then it
slides with linear Z free. Headless, it exits 1 if the table contact was held
for less than CONTACT_HELD of the second half of the slide, the touchdown was faster than
MAX_TOUCHDOWN or the elbow went below MIN_ELBOW_HEIGHT.
"""

from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np
import PyKDL as kdl
import mj_kdl_wrapper as mjk

HOME = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
# [m] TCP start in the base frame, flange down; the base stands on the table, so z is its height.
START_TCP = (0.40, 0.0, 0.12)
TABLE_Z = 0.447
MOVE_X = 0.20
V_MAX_LIN = 0.08  # [m/s] the tracked reference moves toward the target at this speed
V_TOUCH = 0.02  # [m/s] slowest descent, the touchdown speed
DESCENT_GAIN = 0.5  # [1/s] descent speed per metre above the table
KPLIN, KDLIN = 200.0, 30.0
KPROT, KDROT = 175.0, 28.0
BETA_MAX = 120.0
PRESS_FORCE = 10.0  # [N] commanded straight down at the TCP
APPROACH_STEPS = 7500  # no contact within 15 s fails the run
PRESS_STEPS = 250  # the press ramps up over 0.5 s
SLIDE_STEPS = 2000
CONTACT_HELD = 0.6  # the contact chatters while sliding
MAX_TOUCHDOWN = 0.05  # [m/s] TCP vertical speed at first contact
MIN_ELBOW_HEIGHT = 0.30  # [m] above the table
ELBOW_BODY = "forearm_link"  # its origin is the elbow joint
TCP_SITE = "pinch_site"  # the flange face, z out of the arm

APPROACH, PRESS, SLIDE = "approach", "press", "slide"


def build_env() -> tuple[mjk.Env, mjk.Robot]:
    table = mjk.SceneObject()
    table.name = "table"
    table.mjcf_path = mjk.menagerie.asset_path("table.xml", env_var="MJ_KDL_TABLE")
    table.pos = [0.0, 0.0, TABLE_Z]
    table.fixed = True
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    spec.objects = [table]
    robot_spec = mjk.RobotSpec()
    robot_spec.path = mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")
    robot_spec.pos = [0.0, 0.0, TABLE_Z]
    spec.robots = [robot_spec]
    env = mjk.Env.build(spec)
    tool = mjk.ToolFrameSpec()
    tool.tcp_site = TCP_SITE
    robot = env.create_robot("base_link", "bracelet_link", tool=tool)
    return env, robot


def jnt(values) -> kdl.JntArray:
    out = kdl.JntArray(len(values))
    for i, v in enumerate(values):
        out[i] = v
    return out


def solve_near_seed(chain, limits, seed: list[float], target: kdl.Frame) -> list[float]:
    """Damped-least-squares IK stepped from seed, so the answer stays on the seed's branch."""
    fk = kdl.ChainFkSolverPos_recursive(chain)
    ik = kdl.ChainIkSolverVel_wdls(chain, 1e-5, 150)
    ik.setLambda(0.05)
    q = jnt(seed)
    dq = kdl.JntArray(q.rows())
    for _ in range(300):
        current = kdl.Frame()
        fk.JntToCart(q, current)
        dx = kdl.diff(current, target)
        if dx.vel.Norm() <= 2e-3 and dx.rot.Norm() <= 2e-2:
            return [q[i] for i in range(q.rows())]
        if dx.vel.Norm() > 0.05:
            dx.vel = dx.vel * (0.05 / dx.vel.Norm())
        if dx.rot.Norm() > 0.20:
            dx.rot = dx.rot * (0.20 / dx.rot.Norm())
        if ik.CartToJnt(q, dx, dq) < 0:
            raise RuntimeError("PyKDL IK velocity step failed")
        # joint_limits are +-inf for an unlimited joint, so the clamp leaves it alone.
        for i in range(q.rows()):
            q[i] = min(limits[i][1], max(limits[i][0], q[i] + dq[i]))
    current = kdl.Frame()
    fk.JntToCart(q, current)
    dx = kdl.diff(current, target)
    if dx.vel.Norm() > 2e-3 or dx.rot.Norm() > 2e-2:
        raise RuntimeError("PyKDL IK did not converge")
    return [q[i] for i in range(q.rows())]


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


def touches(data: mujoco.MjData, geom: int) -> bool:
    return any(geom in (data.contact[i].geom1, data.contact[i].geom2) for i in range(data.ncon))


def site_sink_speed(model: mujoco.MjModel, data: mujoco.MjData, site: int) -> float:
    """Downward world-frame speed of a site."""
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site, vel, 0)
    return -vel[5]


def alpha_all() -> kdl.Jacobian:
    alpha = kdl.Jacobian(6)
    for i in range(3):
        axis = [0.0, 0.0, 0.0]
        axis[i] = 1.0
        alpha.setColumn(i, kdl.Twist(kdl.Vector(*axis), kdl.Vector.Zero()))
        alpha.setColumn(i + 3, kdl.Twist(kdl.Vector.Zero(), kdl.Vector(*axis)))
    return alpha


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
        root_acc = kdl.Twist(kdl.Vector(0.0, 0.0, -gravity_z), kdl.Vector.Zero())
        self.achd_approach = kdl.ChainHdSolver_Vereshchagin(self.chain, root_acc, 6)
        self.achd = kdl.ChainHdSolver_Vereshchagin(self.chain, root_acc, 5)
        # Driver weights 1 pass the commanded wrench through to the environment.
        self.achd.setDriverWeights(np.ones(5), np.zeros(5))
        self.rnea = kdl.ChainIdSolver_RNE(self.chain, kdl.Vector(0.0, 0.0, gravity_z))
        self.alpha6 = alpha_all()
        self.alpha5 = alpha_no_linear_z()
        segments = self.chain.getNrOfSegments()
        self.f_ext = [kdl.Wrench.Zero() for _ in range(segments)]
        self.no_wrench = [kdl.Wrench.Zero() for _ in range(segments)]
        self.table_geom = env.model.geom("top").id
        self.tcp_site = env.model.site(TCP_SITE).id
        self.restart()
        self.target = kdl.Frame(self.tracked.M, self.tracked.p + kdl.Vector(MOVE_X, 0.0, 0.0))

    def tcp(self) -> kdl.Frame:
        frame = kdl.Frame()
        self.fk.JntToCart(jnt(self.robot.jnt_pos_msr), frame)
        return frame

    def elbow_height(self) -> float:
        return self.env.data.body(ELBOW_BODY).xpos[2] - TABLE_Z

    def restart(self) -> None:
        self.env.update()
        self.tracked = self.tcp()
        self.phase = APPROACH
        self.err_prev = None
        self.steps = 0
        self.touchdown = math.nan  # [m/s] TCP sink speed at first contact
        self.elbow_min = self.elbow_height()
        self.contact_steps = 0
        self.reaction_sum = 0.0
        self.reaction_count = 0

    def control(self) -> None:
        env, robot = self.env, self.robot
        dt = env.model.opt.timestep
        if self.phase == APPROACH:
            # The base stands on the table, so base-frame z is the height above it.
            v = min(max(DESCENT_GAIN * self.tracked.p.z(), V_TOUCH), V_MAX_LIN)
            self.tracked.p -= kdl.Vector(0.0, 0.0, v * dt)
        elif self.phase == SLIDE:
            to_goal = self.target.p - self.tracked.p
            dist = to_goal.Norm()
            if dist > 1e-4:
                self.tracked.p += to_goal * (min(dist, V_MAX_LIN * dt) / dist)

        env.update()
        q, qd = jnt(robot.jnt_pos_msr), jnt(robot.jnt_vel_msr)
        current = kdl.Frame()
        self.fk.JntToCart(q, current)
        err = kdl.diff(current, self.tracked)
        e = [err.vel.x(), err.vel.y(), err.vel.z(), err.rot.x(), err.rot.y(), err.rot.z()]
        prev = self.err_prev or e
        de = [(e[i] - prev[i]) / dt for i in range(6)]
        self.err_prev = e
        gains = [(KPLIN, KDLIN)] * 3 + [(KPROT, KDROT)] * 3
        b = [clamp_abs(kp * e[i] + kd * de[i], BETA_MAX) for i, (kp, kd) in enumerate(gains)]

        n = robot.n_joints
        # Gravity as feed-forward, so only the commanded press pushes the free linear Z down.
        ff = kdl.JntArray(n)
        if self.rnea.CartToJnt(q, kdl.JntArray(n), kdl.JntArray(n), self.no_wrench, ff) < 0:
            raise RuntimeError("PyKDL RNEA failed")
        qdd, constraint_tau, tau = kdl.JntArray(n), kdl.JntArray(n), kdl.JntArray(n)
        if self.phase == APPROACH:
            solver, alpha, beta, f_ext = self.achd_approach, self.alpha6, jnt(b), self.no_wrench
        else:
            ramp = min(1.0, self.steps / PRESS_STEPS) if self.phase == PRESS else 1.0
            press = kdl.Vector(0.0, 0.0, -ramp * PRESS_FORCE)
            self.f_ext[-1] = kdl.Wrench(press, kdl.Vector.Zero())
            solver, alpha, f_ext = self.achd, self.alpha5, self.f_ext
            beta = jnt([b[0], b[1], b[3], b[4], b[5]])
        if solver.CartToJnt(q, qd, qdd, alpha, beta, f_ext, ff, constraint_tau) < 0:
            raise RuntimeError("PyKDL ACHD failed")
        if self.rnea.CartToJnt(q, qd, qdd, self.no_wrench, tau) < 0:
            raise RuntimeError("PyKDL RNEA failed")
        # update() clamps each torque to its joint's limit and reports it in jnt_saturated.
        robot.jnt_trq_cmd = [tau[i] for i in range(n)]
        self.steps += 1

    def measure(self) -> None:
        """Advances the phase after a step and samples the checks."""
        model, data = self.env.model, self.env.data
        self.elbow_min = min(self.elbow_min, self.elbow_height())
        if self.phase == APPROACH and touches(data, self.table_geom):
            self.touchdown = site_sink_speed(model, data, self.tcp_site)
            self.target = kdl.Frame(self.tracked.M, self.tracked.p + kdl.Vector(MOVE_X, 0.0, 0.0))
            self.phase, self.steps = PRESS, 0
        elif self.phase == PRESS and self.steps >= PRESS_STEPS:
            self.phase, self.steps = SLIDE, 0
        # Measured over the second half of the slide: the contact chatters while the TCP moves.
        elif self.phase == SLIDE and self.steps > SLIDE_STEPS // 2:
            reaction = contact_normal_force(model, data, self.table_geom)
            self.contact_steps += reaction > 0.0
            self.reaction_sum += reaction
            self.reaction_count += 1

    def done(self) -> bool:
        return (self.phase == APPROACH and self.steps >= APPROACH_STEPS) or self.finished()

    def finished(self) -> bool:
        return self.phase == SLIDE and self.steps >= SLIDE_STEPS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    env, robot = build_env()
    try:
        restarted = [False]
        start = kdl.Frame(kdl.Rotation.RotX(math.pi), kdl.Vector(*START_TCP))
        q_start = solve_near_seed(robot.kdl_chain(), robot.joint_limits, HOME, start)

        def on_reset(ctx):
            robot.set_joint_pos(q_start)
            robot.jnt_pos_cmd = q_start
            restarted[0] = True

        env.on_reset = on_reset
        env.reset()
        robot.set_control_mode(mjk.CtrlMode.TORQUE)
        slide = Slide(env, robot)

        if args.gui:
            # The UI's reset button runs env's reset, on_reset included.
            env.open_viewer("ex_achd_table_slide.py")
        restarted[0] = False
        while not slide.done():
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
        print(
            f"touchdown_speed_m_s={slide.touchdown:.4f} (limit {MAX_TOUCHDOWN:.4f}) "
            f"elbow_min_z_above_table={slide.elbow_min:.4f} (limit {MIN_ELBOW_HEIGHT:.4f})"
        )
    finally:
        env.close()
    if args.gui:
        return 0
    ok = (
        slide.finished()
        and contact >= CONTACT_HELD
        and slide.touchdown <= MAX_TOUCHDOWN
        and slide.elbow_min >= MIN_ELBOW_HEIGHT
    )
    goal = "the TCP touched down gently and held the table during the slide"
    print(f"{'PASS' if ok else 'FAIL'}: {goal}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
