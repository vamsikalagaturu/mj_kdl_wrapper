#!/usr/bin/env python3
"""Table pick-place with ACHD + RNEA; Python counterpart of src/examples/ex_achd_pick_place.cpp."""

from __future__ import annotations

import argparse
import math

import PyKDL as kdl

import mj_kdl_wrapper as mjk

HOME = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
SURFACE_Z = 0.70
CUBE_HS = 0.02
PICK = (0.40, 0.00)
PLACE = (0.40, 0.24)
CUBE_START = [PICK[0], PICK[1], SURFACE_Z + CUBE_HS]
GRIPPER_CLOSED = 0.82  # [rad] top of the 2F-85's ctrlrange (its driver joint stops at 0.8)
MAX_PLACE_ERR = 0.005  # [m] in the table plane
MAX_ELBOW_DROP = 0.10  # [m] below the support reference

# With alpha = I_6 these are the desired TCP linear/angular accelerations.
KP_LIN, KI_LIN, KD_LIN = 200.0, 100.0, 40.0
KP_ROT, KI_ROT, KD_ROT = 120.0, 50.0, 80.0
BETA_LIN_MAX, BETA_ROT_MAX = 120.0, 80.0
INTEGRAL_MAX = 0.5

# An ACHD-only upward wrench on this link keeps the elbow from dropping in every phase.
SUPPORT_LINK = "half_arm_2_link"
SUPPORT_KP, SUPPORT_KD, SUPPORT_F_MAX, SUPPORT_LIFT = 800.0, 80.0, 45.0, 0.06


class ResetRequested(Exception):
    """Raised after a simulate UI reset, to restart the sequence."""


def build_env() -> tuple[mjk.Env, mjk.Robot]:
    table = mjk.SceneObject()
    table.name = "table"
    table.mjcf_path = mjk.menagerie.asset_path("table.xml", env_var="MJ_KDL_TABLE")
    table.pos = [0.0, 0.0, SURFACE_Z]
    table.fixed = True
    cube = mjk.SceneObject()
    cube.name = "cube"
    cube.shape = mjk.Shape.BOX
    cube.size = [CUBE_HS, CUBE_HS, CUBE_HS]
    cube.pos = CUBE_START[:]
    cube.rgba = [0.1, 0.35, 1.0, 1.0]
    cube.mass = 0.1
    cube.condim = mjk.Condim.Torsional
    cube.friction = [0.8, 0.02, 0.001]
    gripper = mjk.AttachmentSpec()
    gripper.mjcf_path = mjk.menagerie.asset_path("robotiq_2f85/2f85.xml", env_var="MJ_KDL_GRIPPER")
    gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
    gripper.prefix = "g_"
    robot_spec = mjk.RobotSpec()
    robot_spec.path = mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")
    robot_spec.pos = [0.0, 0.0, SURFACE_Z]
    robot_spec.attachments = [gripper]

    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    spec.objects = [table, cube]
    spec.robots = [robot_spec]
    env = mjk.Env.build(spec)
    tool = mjk.ToolFrameSpec()
    tool.tool_body = "g_base_mount"
    tool.tcp_site = "g_pinch"
    return env, env.create_robot("base_link", "bracelet_link", tool=tool)


def jnt(values) -> kdl.JntArray:
    out = kdl.JntArray(len(values))
    for i, v in enumerate(values):
        out[i] = v
    return out


def clamp_abs(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def smoothstep(value: float) -> float:
    t = max(0.0, min(1.0, value))
    return t * t * (3.0 - 2.0 * t)


class Achd:
    """ACHD qddot for a TCP target, priced by RNEA into the robot's torque command."""

    def __init__(self, robot: mjk.Robot, gravity_z: float, dt: float):
        self.robot, self.dt = robot, dt
        # KDL solvers keep a reference to the chain, so it must outlive them.
        self.chain = chain = robot.kdl_chain()
        self.n, self.n_seg = robot.n_joints, chain.getNrOfSegments()
        self.fk = kdl.ChainFkSolverPos_recursive(chain)
        self.fk_vel = kdl.ChainFkSolverVel_recursive(chain)
        root_acc = kdl.Twist(kdl.Vector(0.0, 0.0, -gravity_z), kdl.Vector.Zero())
        self.achd = kdl.ChainHdSolver_Vereshchagin(chain, root_acc, 6)
        self.rnea = kdl.ChainIdSolver_RNE(chain, kdl.Vector(0.0, 0.0, gravity_z))
        self.alpha = kdl.Jacobian(6)
        for i in range(6):
            self.alpha[i, i] = 1.0
        self.f_zero = [kdl.Wrench.Zero() for _ in range(self.n_seg)]
        self.support = -1
        for i in range(self.n_seg):
            if chain.getSegment(i).getName() == SUPPORT_LINK:
                self.support = i
        if self.support < 0:
            raise RuntimeError(f"support segment not found: {SUPPORT_LINK}")
        self.new_phase()

    def new_phase(self) -> None:
        self.err_i = [0.0] * 6
        self.err_prev = None

    def measure(self) -> kdl.Frame:
        self.q, self.qd = jnt(self.robot.jnt_pos_msr), jnt(self.robot.jnt_vel_msr)
        self.tcp = kdl.Frame()
        self.fk.JntToCart(self.q, self.tcp)
        return self.tcp

    def link_z(self) -> float:
        frame = kdl.Frame()
        self.fk.JntToCart(self.q, frame, self.support + 1)
        return SURFACE_Z + frame.p.z()

    def control(self, target: kdl.Frame, target_twist: kdl.Twist, f_ext: list) -> None:
        tcp_vel = kdl.FrameVel()
        self.fk_vel.JntToCart(kdl.JntArrayVel(self.q, self.qd), tcp_vel)
        w = tcp_vel.deriv()
        err = kdl.diff(self.tcp, target)
        dt = self.dt
        e = [err.vel.x(), err.vel.y(), err.vel.z(), err.rot.x(), err.rot.y(), err.rot.z()]
        prev = self.err_prev or e
        self.err_prev = e
        # The rotation is damped against the measured rate: the target's own turn is no error.
        d_rot = [
            target_twist.rot.x() - w.rot.x(),
            target_twist.rot.y() - w.rot.y(),
            target_twist.rot.z() - w.rot.z(),
        ]
        beta = kdl.JntArray(6)
        for i in range(6):
            self.err_i[i] = clamp_abs(self.err_i[i] + e[i] * dt, INTEGRAL_MAX)
            if i < 3:
                accel = KP_LIN * e[i] + KI_LIN * self.err_i[i] + KD_LIN * (e[i] - prev[i]) / dt
                beta[i] = clamp_abs(accel, BETA_LIN_MAX)
            else:
                accel = KP_ROT * e[i] + KI_ROT * self.err_i[i] + KD_ROT * d_rot[i - 3]
                beta[i] = clamp_abs(accel, BETA_ROT_MAX)
        qdd, ff, constraint_tau, tau = (kdl.JntArray(self.n) for _ in range(4))
        q, qd, alpha = self.q, self.qd, self.alpha
        if self.achd.CartToJnt(q, qd, qdd, alpha, beta, f_ext, ff, constraint_tau) < 0:
            raise RuntimeError("PyKDL ACHD failed")
        if self.rnea.CartToJnt(self.q, self.qd, qdd, self.f_zero, tau) < 0:
            raise RuntimeError("PyKDL RNEA failed")
        self.robot.jnt_trq_cmd = [tau[i] for i in range(self.n)]


def run_phase(env, ctrl: Achd, fingers, phase: dict, state: dict) -> bool:
    """One update() per step: it reads the state and applies the previous cycle's command."""
    print(f"State: {phase['name']}")
    t0 = env.data.time
    phase_start = ctrl.measure()
    support = state["support"]
    if phase["name"] == "PLACE_ABOVE" or not support["valid"]:
        support.update(z_ref=ctrl.link_z() + SUPPORT_LIFT, valid=True)
        support["prev_z"] = support["z_ref"]
    ctrl.new_phase()
    prev_target = phase_start
    while True:
        env.update()
        tcp = ctrl.measure()
        elapsed = env.data.time - t0
        s = smoothstep(elapsed / phase["duration"]) if phase["duration"] > 0.0 else 1.0
        target = kdl.addDelta(phase_start, kdl.diff(phase_start, phase["target"]), s)
        target_twist = kdl.diff(prev_target, target, ctrl.dt)
        prev_target = target

        f_ext = [kdl.Wrench.Zero() for _ in range(ctrl.n_seg)]
        z = ctrl.link_z()
        vz = (z - support["prev_z"]) / ctrl.dt
        support["prev_z"] = z
        support["drop"] = max(support["drop"], support["z_ref"] - SUPPORT_LIFT - z)
        fz = min(SUPPORT_F_MAX, max(0.0, SUPPORT_KP * (support["z_ref"] - z) - SUPPORT_KD * vz))
        f_ext[ctrl.support] = kdl.Wrench(kdl.Vector(0.0, 0.0, fz), kdl.Vector.Zero())
        ctrl.control(target, target_twist, f_ext)
        fingers.ctrl[0] = phase["gripper"]

        err = kdl.diff(tcp, phase["target"])
        settled = phase["pos_tol"] < 0.0 or (
            err.vel.Norm() <= phase["pos_tol"] and err.rot.Norm() <= phase["rot_tol"]
        )
        if (elapsed >= phase["duration"] and settled) or elapsed >= phase["timeout"]:
            return True
        if not env.step():
            return False
        if state["reset"]:
            state["reset"] = False
            raise ResetRequested()
        env.pace()


def build_phases(robot: mjk.Robot, home_tcp: kdl.Frame) -> list[dict]:
    grasp_rot = robot.tip_T_tcp.M
    z_grasp = CUBE_HS
    z_above = z_grasp + 0.20
    z_lift = z_grasp + 0.30

    def at(xy, z: float) -> kdl.Frame:
        return kdl.Frame(grasp_rot, kdl.Vector(xy[0], xy[1], z))

    def phase(name, target, duration, timeout, pos_tol, rot_tol, gripper) -> dict:
        return dict(
            name=name,
            target=target,
            duration=duration,
            timeout=timeout,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            gripper=gripper,
        )

    closed = GRIPPER_CLOSED
    return [
        phase("HOME", home_tcp, 1.0, 2.5, 0.03, 0.05, 0.0),
        phase("PICK_ABOVE", at(PICK, z_above), 8.0, 14.0, 0.04, 0.03, 0.0),
        phase("PICK", at(PICK, z_grasp), 5.0, 12.0, 0.02, 0.03, 0.0),
        phase("CLOSE", at(PICK, z_grasp), 1.5, 2.5, -1.0, -1.0, closed),
        phase("LIFT", at(PICK, z_lift), 3.0, 8.0, 0.04, 0.03, closed),
        phase("PLACE_ABOVE", at(PLACE, z_above), 5.0, 12.0, 0.04, 0.03, closed),
        phase("PLACE", at(PLACE, z_grasp), 5.0, 14.0, 0.02, 0.03, closed),
        phase("OPEN", at(PLACE, z_grasp), 1.0, 2.0, -1.0, -1.0, 0.0),
        phase("RETREAT", at(PLACE, z_above), 3.0, 6.0, 0.04, 0.08, 0.0),
        phase("HOLD", at(PLACE, z_above), 4.0, 4.0, -1.0, -1.0, 0.0),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    env, robot = build_env()
    try:
        robot.set_control_mode(mjk.CtrlMode.TORQUE)
        fingers = env.data.actuator("g_fingers_actuator")
        gravity_z = env.spec.gravity_z
        ctrl = Achd(robot, gravity_z, env.model.opt.timestep)
        dyn = kdl.ChainDynParam(ctrl.chain, kdl.Vector(0.0, 0.0, gravity_z))
        home_tcp = kdl.Frame()
        ctrl.fk.JntToCart(jnt(HOME), home_tcp)
        phases = build_phases(robot, home_tcp)
        state = {"reset": False}

        def on_reset(ctx):
            robot.set_joint_pos(HOME)
            env.set_body_pose("cube", CUBE_START)
            fingers.ctrl[0] = 0.0
            g = kdl.JntArray(robot.n_joints)
            dyn.JntToGravity(jnt(HOME), g)
            robot.jnt_trq_cmd = [g[i] for i in range(robot.n_joints)]
            state["reset"] = True

        env.on_reset = on_reset
        env.reset()
        if args.gui:
            env.open_viewer("ex_achd_pick_place.py")

        completed = False
        while not completed:
            state.update(
                reset=False, support={"valid": False, "z_ref": 0.0, "prev_z": 0.0, "drop": 0.0}
            )
            try:
                completed = all(run_phase(env, ctrl, fingers, p, state) for p in phases)
                break
            except ResetRequested:
                continue
        env.update()

        cube = env.body_frame("cube").p
        place_err = math.hypot(cube.x() - PLACE[0], cube.y() - PLACE[1])
        on_table = abs(cube.z() - (SURFACE_Z + CUBE_HS)) < 0.002
        print(
            f"cube final position: [{cube.x():.4f}, {cube.y():.4f}, {cube.z():.4f}] "
            f"place error {place_err * 1000:.4f} mm (limit {MAX_PLACE_ERR * 1000} mm)"
            + ("" if on_table else ", not on the table")
        )
        drop = state["support"]["drop"]
        print(f"elbow drop: {drop * 1000:.4f} mm (limit {MAX_ELBOW_DROP * 1000} mm)")
    finally:
        env.close()
    if args.gui:
        return 0
    ok = completed and on_table and place_err <= MAX_PLACE_ERR and drop <= MAX_ELBOW_DROP
    print(f"{'PASS' if ok else 'FAIL'}: the cube was placed with the elbow held up")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
