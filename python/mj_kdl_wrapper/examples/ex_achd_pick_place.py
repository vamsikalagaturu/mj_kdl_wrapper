#!/usr/bin/env python3
"""ACHD Cartesian pick-place example using PyKDL directly.

Ported from src/examples/ex_achd_pick_place.cpp. Uses the Vereshchagin
acceleration-constrained hybrid-dynamics solver plus RNEA for computed torques.
An Env on_reset hook re-homes the arm, re-poses the cube, and opens the gripper.
"""

from __future__ import annotations

import argparse
import math

import PyKDL as kdl
import mj_kdl_wrapper as mjk

HOME = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
SURFACE_Z = 0.70
CUBE_HS = 0.02
PICK_X, PICK_Y = 0.40, 0.00
PLACE_X, PLACE_Y = 0.40, 0.24
CUBE_START = [PICK_X, PICK_Y, SURFACE_Z + CUBE_HS]

# With alpha = I_6 the PID's output is the desired TCP acceleration, which ACHD consumes.
KP_LIN, KI_LIN, KD_LIN = 200.0, 100.0, 40.0
KP_ROT, KI_ROT, KD_ROT = 120.0, 50.0, 80.0
BETA_LIN_MAX, BETA_ROT_MAX = 120.0, 80.0
INTEGRAL_MAX, TAU_MAX = 0.5, 59.0

# Rides the 7th joint's redundancy: keeps the arm from folding near the table.
SUPPORT_LINK = "half_arm_2_link"
SUPPORT_KP, SUPPORT_KD, SUPPORT_F_MAX, SUPPORT_LIFT = 800.0, 80.0, 45.0, 0.06
SUPPORT_PHASES = ("PLACE_ABOVE", "PLACE", "OPEN", "RETREAT")

GRIPPER_CLOSED = 0.8  # the 2f85 actuator's ctrlrange is 0 to 0.82


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
    cube.size = [CUBE_HS, CUBE_HS, CUBE_HS]
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


def segment_index(chain: kdl.Chain, name: str) -> int:
    for i in range(chain.getNrOfSegments()):
        if chain.getSegment(i).getName() == name:
            return i
    return -1


def link_world_z(fk: kdl.ChainFkSolverPos_recursive, q: kdl.JntArray, index: int) -> float:
    frame = kdl.Frame()
    fk.JntToCart(q, frame, index + 1)
    return SURFACE_Z + frame.p.z()


def support_wrench(z_world: float, z_ref: float, vz: float) -> kdl.Wrench:
    fz = min(SUPPORT_F_MAX, max(0.0, SUPPORT_KP * (z_ref - z_world) - SUPPORT_KD * vz))
    return kdl.Wrench(kdl.Vector(0.0, 0.0, fz), kdl.Vector.Zero())


def achd_step(ctx, target: kdl.Frame, target_twist: kdl.Twist, f_ext: list, state: dict) -> None:
    robot = ctx["robot"]
    ctx["env"].update()
    n = robot.n_joints
    q = jnt(robot.jnt_pos_msr)
    qd = jnt(robot.jnt_vel_msr)

    current = kdl.Frame()
    ctx["fk"].JntToCart(q, current)
    current_vel = kdl.FrameVel()
    ctx["fk_vel"].JntToCart(kdl.JntArrayVel(q, qd), current_vel)
    tcp_twist = current_vel.deriv()

    err = kdl.diff(current, target)
    dt = ctx["dt"]
    e = [err.vel.x(), err.vel.y(), err.vel.z(), err.rot.x(), err.rot.y(), err.rot.z()]
    if state["first"]:
        state["err_prev"] = e[:]
        state["first"] = False
    de = [(e[i] - state["err_prev"][i]) / dt for i in range(6)]
    state["err_prev"] = e[:]
    ei = state["err_i"]
    for i in range(6):
        ei[i] = clamp_abs(ei[i] + e[i] * dt, INTEGRAL_MAX)

    # Damped against the measured angular velocity: the target's own rotation is not an error rate.
    d_rot = [
        target_twist.rot.x() - tcp_twist.rot.x(),
        target_twist.rot.y() - tcp_twist.rot.y(),
        target_twist.rot.z() - tcp_twist.rot.z(),
    ]
    beta = kdl.JntArray(6)
    for i in range(3):
        beta[i] = clamp_abs(KP_LIN * e[i] + KI_LIN * ei[i] + KD_LIN * de[i], BETA_LIN_MAX)
    for i in range(3):
        beta[3 + i] = clamp_abs(
            KP_ROT * e[3 + i] + KI_ROT * ei[3 + i] + KD_ROT * d_rot[i], BETA_ROT_MAX
        )

    qdd = kdl.JntArray(n)
    ff = kdl.JntArray(n)
    constraint_tau = kdl.JntArray(n)
    if ctx["achd"].CartToJnt(q, qd, qdd, ctx["alpha"], beta, f_ext, ff, constraint_tau) < 0:
        raise RuntimeError("PyKDL ACHD failed")

    # RNEA re-prices the resolved acceleration for MuJoCo, and never sees the wrench.
    tau = kdl.JntArray(n)
    if ctx["rnea"].CartToJnt(q, qd, qdd, ctx["f_ext_zero"], tau) < 0:
        raise RuntimeError("PyKDL RNEA failed")
    robot.jnt_trq_cmd = [clamp_abs(tau[i], TAU_MAX) for i in range(n)]
    ctx["env"].update()


# The UI's reset button has already reset env (on_reset included); restart the phases.
def step_once(env, gui, state) -> bool:
    if not env.step():
        return False
    if gui and env.time() < state["prev"] - 1e-6:
        state["prev"] = env.time()
        raise ResetRequested()
    state["prev"] = env.time()
    return True


def run_phase(ctx, phase: dict, gui, state) -> bool:
    print(f"State: {phase['name']}")
    env, robot = ctx["env"], ctx["robot"]
    env.update()
    phase_start = robot.fk_frame()
    t0 = env.time()
    pid = {"err_prev": [0.0] * 6, "err_i": [0.0] * 6, "first": True}
    prev_target = phase_start
    first_target = True
    support = state["support"]
    if phase["name"] == "PLACE_ABOVE" or (phase["name"] in SUPPORT_PHASES and not support["valid"]):
        q = jnt(robot.jnt_pos_msr)
        support["z_ref"] = link_world_z(ctx["fk"], q, ctx["support_segment"]) + SUPPORT_LIFT
        support["prev_z"] = support["z_ref"]
        support["valid"] = True

    while True:
        elapsed = env.time() - t0
        alpha_t = smoothstep(elapsed / phase["duration"]) if phase["duration"] > 0.0 else 1.0
        target = kdl.addDelta(phase_start, kdl.diff(phase_start, phase["target"]), alpha_t)
        target_twist = kdl.Twist.Zero()
        if not first_target:
            target_twist = kdl.diff(prev_target, target, ctx["dt"])
        prev_target = target
        first_target = False

        f_ext = [kdl.Wrench.Zero() for _ in range(ctx["n_segments"])]
        if phase["name"] in SUPPORT_PHASES and support["valid"]:
            q = jnt(robot.jnt_pos_msr)
            z_world = link_world_z(ctx["fk"], q, ctx["support_segment"])
            vz = (z_world - support["prev_z"]) / ctx["dt"]
            support["prev_z"] = z_world
            f_ext[ctx["support_segment"]] = support_wrench(z_world, support["z_ref"], vz)

        if env.has_actuator("g_fingers_actuator"):
            env.set_actuator_ctrl("g_fingers_actuator", phase["gripper"])
        achd_step(ctx, target, target_twist, f_ext, pid)

        err = kdl.diff(robot.fk_frame(), phase["target"])
        settled = phase["pos_tol"] < 0.0 or (
            err.vel.Norm() <= phase["pos_tol"] and err.rot.Norm() <= phase["rot_tol"]
        )
        if (elapsed >= phase["duration"] and settled) or elapsed >= phase["timeout"]:
            return True
        if gui and not env.viewer.is_running():
            return False
        if not step_once(env, gui, state):
            return False


def build_phases(robot, chain) -> list[dict]:
    # The grasp orientation is the tool frame's own: the pinch axis points at the table.
    grasp_rot = robot.tip_to_tcp.M
    z_grasp = CUBE_HS
    z_above = z_grasp + 0.20
    z_lift = z_grasp + 0.30

    def at(x: float, y: float, z: float) -> kdl.Frame:
        return kdl.Frame(grasp_rot, kdl.Vector(x, y, z))

    def phase(name, target, duration, timeout, pos_tol, rot_tol, gripper) -> dict:
        return {
            "name": name,
            "target": target,
            "duration": duration,
            "timeout": timeout,
            "pos_tol": pos_tol,
            "rot_tol": rot_tol,
            "gripper": gripper,
        }

    closed = GRIPPER_CLOSED
    return [
        phase("HOME", robot.fk_frame(), 1.0, 2.5, 0.03, 0.05, 0.0),
        phase("PICK_ABOVE", at(PICK_X, PICK_Y, z_above), 8.0, 14.0, 0.04, 0.03, 0.0),
        phase("PICK", at(PICK_X, PICK_Y, z_grasp), 5.0, 12.0, 0.02, 0.03, 0.0),
        phase("CLOSE", at(PICK_X, PICK_Y, z_grasp), 1.5, 2.5, -1.0, -1.0, closed),
        phase("LIFT", at(PICK_X, PICK_Y, z_lift), 3.0, 8.0, 0.04, 0.03, closed),
        phase("PLACE_ABOVE", at(PLACE_X, PLACE_Y, z_above), 5.0, 12.0, 0.04, 0.03, closed),
        phase("PLACE", at(PLACE_X, PLACE_Y, z_grasp), 5.0, 14.0, 0.02, 0.03, closed),
        phase("OPEN", at(PLACE_X, PLACE_Y, z_grasp), 1.0, 2.0, -1.0, -1.0, 0.0),
        phase("RETREAT", at(PLACE_X, PLACE_Y, z_above), 3.0, 6.0, 0.04, 0.08, 0.0),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    env, robot = build_env()
    try:
        chain = robot.kdl_chain()
        support_seg = segment_index(chain, SUPPORT_LINK)
        if support_seg < 0:
            raise RuntimeError(f"support segment not found: {SUPPORT_LINK}")

        ctx = {
            "env": env,
            "robot": robot,
            "dt": env.timestep(),
            "n_segments": chain.getNrOfSegments(),
            "support_segment": support_seg,
            "fk": kdl.ChainFkSolverPos_recursive(chain),
            "fk_vel": kdl.ChainFkSolverVel_recursive(chain),
            "achd": kdl.ChainHdSolver_Vereshchagin(
                chain, kdl.Twist(kdl.Vector(0.0, 0.0, 9.81), kdl.Vector.Zero()), 6
            ),
            "rnea": kdl.ChainIdSolver_RNE(chain, kdl.Vector(0.0, 0.0, -9.81)),
            "alpha": alpha_identity(),
            "f_ext_zero": [kdl.Wrench.Zero() for _ in range(chain.getNrOfSegments())],
        }

        robot.set_control_mode(mjk.CtrlMode.TORQUE)

        def on_reset(ctx_unused):
            robot.set_joint_pos(HOME)
            env.set_body_pose("cube", CUBE_START)
            if env.has_actuator("g_fingers_actuator"):
                env.set_actuator_ctrl("g_fingers_actuator", 0.0)

        env.on_reset = on_reset
        env.reset()
        env.update()
        phases = build_phases(robot, chain)

        state = {"prev": env.time(), "support": {"valid": False, "z_ref": 0.0, "prev_z": 0.0}}
        if args.gui:
            env.open_viewer("ex_achd_pick_place.py")
            while env.viewer.is_running():
                try:
                    for phase in phases:
                        if not run_phase(ctx, phase, True, state):
                            raise StopIteration
                    break
                except ResetRequested:
                    state["support"] = {"valid": False, "z_ref": 0.0, "prev_z": 0.0}
                    continue
                except StopIteration:
                    break
        else:
            for phase in phases:
                if not run_phase(ctx, phase, False, state):
                    break

        cube = env.body_frame("cube")
        error_xy = math.hypot(cube.p.x() - PLACE_X, cube.p.y() - PLACE_Y)
        print(
            f"cube final position: [{cube.p.x():.3f}, {cube.p.y():.3f}, {cube.p.z():.3f}]"
            f" target=[{PLACE_X:.3f}, {PLACE_Y:.3f}, {SURFACE_Z + CUBE_HS:.3f}]"
            f" xy_error={error_xy:.3f}"
        )
        if not args.gui and error_xy > 0.08:
            return 1
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
