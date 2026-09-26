#!/usr/bin/env python3
"""Table pick-place with joint impedance, pushed while carrying the cube; Python counterpart
of ex_table_pick_place.cpp. Headless it exits 1 unless the cube is placed, the push deflection
matches J K^-1 J^T F, the arm springs back, the cube stays in hand and the elbow stays up."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import PyKDL as kdl

import mj_kdl_wrapper as mjk

HOME = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
SURFACE_Z = 0.70
CUBE_HS = 0.02
PICK = (0.40, 0.00)
PLACE = (0.40, 0.24)
KP = [100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0]
KD = [10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0]
GRIPPER_CLOSED = 0.82  # [rad] top of the 2F-85's ctrlrange (its driver joint stops at 0.8)
CUBE_START = [PICK[0], PICK[1], SURFACE_Z + CUBE_HS]
MAX_PLACE_ERR = 0.005  # [m] in the table plane

PUSH_BODY = "bracelet_link"
PUSH_PHASE = "PLACE_ABOVE"
PUSH_FORCE = 20.0  # [N] along world +x, across the carry along +y
PUSH_DIR = kdl.Vector(1.0, 0.0, 0.0)
PUSH_ON, PUSH_OFF, PUSH_RAMP = 0.8, 2.0, 0.2  # [s] into the phase; rise and fall
SETTLE = 0.5  # [s] after the push, when the residual is taken

MIN_DEFLECTION = 0.010  # [m]
DEFLECTION_RATIO = (0.85, 1.15)  # measured / predicted
MAX_RESIDUAL = 0.004  # [m]
MAX_CUBE_SLIP = 0.010  # [m] cube centre from the TCP while pushed
MIN_ELBOW_HEIGHT = 0.45  # [m] forearm_link origin above the table


class ResetRequested(Exception):
    """Raised after a simulate UI reset, to restart the sequence."""


@dataclass(frozen=True)
class Phase:
    name: str
    target: list[float]
    duration: float
    timeout: float
    settle_tol: float
    gripper: float


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
    q = kdl.JntArray(len(values))
    for i, value in enumerate(values):
        q[i] = value
    return q


def gravity(dyn: kdl.ChainDynParam, q_values) -> list[float]:
    g = kdl.JntArray(len(q_values))
    dyn.JntToGravity(jnt(q_values), g)
    return [g[i] for i in range(g.rows())]


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
            break
        if dx.vel.Norm() > 0.05:
            dx.vel = dx.vel * (0.05 / dx.vel.Norm())
        if dx.rot.Norm() > 0.20:
            dx.rot = dx.rot * (0.20 / dx.rot.Norm())
        if ik.CartToJnt(q, dx, dq) < 0:
            raise RuntimeError("PyKDL IK velocity step failed")
        # joint_limits are +-inf for an unlimited joint, so the clamp leaves it alone.
        for i in range(q.rows()):
            q[i] = min(limits[i][1], max(limits[i][0], q[i] + dq[i]))
    else:
        current = kdl.Frame()
        fk.JntToCart(q, current)
        dx = kdl.diff(current, target)
        if dx.vel.Norm() > 2e-3 or dx.rot.Norm() > 2e-2:
            raise RuntimeError("PyKDL IK did not converge")
    return [q[i] for i in range(q.rows())]


def pick_place_phases(robot: mjk.Robot) -> list[Phase]:
    chain = robot.kdl_chain()
    seed = HOME[:]

    # Base-frame heights: the TCP at the cube centre, then above it.
    def solve(xy, z: float) -> list[float]:
        nonlocal seed
        target = kdl.Frame(robot.tip_T_tcp.M, kdl.Vector(xy[0], xy[1], z))
        seed = solve_near_seed(chain, robot.joint_limits, seed, target)
        return seed[:]

    pick_above = solve(PICK, CUBE_HS + 0.20)
    pick = solve(PICK, CUBE_HS)
    lift = solve(PICK, CUBE_HS + 0.30)
    place_above = solve(PLACE, CUBE_HS + 0.20)
    place = solve(PLACE, CUBE_HS)
    closed = GRIPPER_CLOSED
    return [
        Phase("HOME", HOME, 1.0, 2.5, 0.08, 0.0),
        Phase("PICK_ABOVE", pick_above, 5.0, 7.0, 0.08, 0.0),
        Phase("PICK", pick, 5.0, 8.0, 0.03, 0.0),
        Phase("CLOSE", pick, 1.5, 2.5, -1.0, closed),
        Phase("LIFT", lift, 3.0, 5.0, 0.08, closed),
        Phase("PLACE_ABOVE", place_above, 3.0, 5.0, 0.08, closed),
        Phase("PLACE", place, 5.0, 8.0, 0.03, closed),
        Phase("OPEN", place, 1.0, 2.0, -1.0, 0.0),
        Phase("RETREAT", place_above, 2.0, 4.0, 0.08, 0.0),
        Phase("HOLD", place_above, 1.0, 1.0, -1.0, 0.0),
    ]


def new_metrics() -> dict:
    return {
        "lag0": kdl.Vector(),  # TCP minus its reference just before the push
        "peak": 0.0,  # [m] along the push
        "predicted": 0.0,  # [m] J K^-1 J^T F at the peak
        "residual": -1.0,  # [m] along the push, SETTLE after it; < 0 until taken
        "cube_slip": 0.0,
        "elbow_min": math.inf,  # [m] world z
    }


def push_scale(t_rel: float) -> float:
    rise = (t_rel - PUSH_ON) / PUSH_RAMP
    fall = (PUSH_OFF - t_rel) / PUSH_RAMP
    return min(1.0, max(0.0, min(rise, fall)))


def measure_push(env, robot, state: dict, q_ref, cube, t_rel: float, force) -> None:
    """Deflection against the reference of the same cycle, and what the stiffness predicts."""
    m, fk, jac_solver = state["metrics"], state["fk"], state["jac"]
    base = kdl.Vector(0.0, 0.0, SURFACE_Z)
    tcp, ref = kdl.Frame(), kdl.Frame()
    fk.JntToCart(jnt(robot.jnt_pos_msr), tcp)
    fk.JntToCart(jnt(q_ref), ref)
    m["cube_slip"] = max(m["cube_slip"], (cube - base - tcp.p).Norm())
    lag = tcp.p - ref.p
    if t_rel < PUSH_ON:
        m["lag0"] = lag
    deflection = kdl.dot(lag - m["lag0"], PUSH_DIR)
    if t_rel >= PUSH_OFF + SETTLE and m["residual"] < 0.0:
        m["residual"] = abs(deflection)
    # Only at full force: on the ramps the arm lags the changing force.
    full = PUSH_ON + PUSH_RAMP <= t_rel <= PUSH_OFF - PUSH_RAMP
    if not full or deflection <= m["peak"]:
        return

    # dq = K^-1 J_push^T F at the pushed body's centre of mass, seen at the TCP through J.
    n = robot.n_joints
    jac = kdl.Jacobian(n)
    jac_solver.JntToJac(jnt(q_ref), jac)
    jac_push = kdl.Jacobian(jac)
    jac_push.changeRefPoint(kdl.Vector(*env.data.body(PUSH_BODY).xipos) - base - ref.p)
    dx = kdl.Vector()
    for i in range(n):
        dx += jac.getColumn(i).vel * (kdl.dot(jac_push.getColumn(i).vel, force) / KP[i])
    m["peak"] = deflection
    m["predicted"] = kdl.dot(dx, PUSH_DIR)


def run_phase(env, robot, fingers, dyn, phase: Phase, state: dict) -> bool:
    """One update() per step: it reads the state and applies the previous cycle's command."""
    print(f"State: {phase.name}")
    start = robot.jnt_pos_msr[:]
    t0 = env.data.time
    push = env.data.body(PUSH_BODY)
    pushing = phase.name == PUSH_PHASE
    while True:
        env.update()
        elapsed = env.data.time - t0
        alpha = min(1.0, max(0.0, elapsed / phase.duration)) if phase.duration > 0.0 else 1.0
        q, qd = robot.jnt_pos_msr, robot.jnt_vel_msr
        q_ref = [start[i] + alpha * (phase.target[i] - start[i]) for i in range(len(q))]
        g = gravity(dyn, q)
        robot.jnt_trq_cmd = [
            g[i] + KP[i] * (q_ref[i] - q[i]) - KD[i] * qd[i] for i in range(len(q))
        ]
        fingers.ctrl[0] = phase.gripper
        cube = env.body_frame("cube").p if pushing else None
        err = max(abs(t - m) for t, m in zip(phase.target, q))
        done_pose = phase.settle_tol < 0.0 or err <= phase.settle_tol
        if (elapsed >= phase.duration and done_pose) or elapsed >= phase.timeout:
            return True
        if not env.step():
            return False
        if state["reset"]:
            state["reset"] = False
            raise ResetRequested()
        env.pace()
        metrics = state["metrics"]
        metrics["elbow_min"] = min(metrics["elbow_min"], env.body_frame("forearm_link").p.z())
        force = PUSH_DIR * (PUSH_FORCE * push_scale(elapsed) if pushing else 0.0)
        push.xfrc_applied[:] = [force.x(), force.y(), force.z(), 0.0, 0.0, 0.0]
        if pushing:
            measure_push(env, robot, state, q_ref, cube, elapsed, force)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    env, robot = build_env()
    try:
        robot.set_control_mode(mjk.CtrlMode.TORQUE)
        fingers = env.data.actuator("g_fingers_actuator")
        chain = robot.kdl_chain()
        dyn = kdl.ChainDynParam(chain, kdl.Vector(0.0, 0.0, env.spec.gravity_z))
        phases = pick_place_phases(robot)
        state = {
            "reset": False,
            "metrics": new_metrics(),
            "fk": kdl.ChainFkSolverPos_recursive(chain),
            "jac": kdl.ChainJntToJacSolver(chain),
        }

        def on_reset(ctx):
            robot.set_joint_pos(HOME)
            env.set_body_pose("cube", CUBE_START)
            fingers.ctrl[0] = 0.0
            robot.jnt_trq_cmd = gravity(dyn, HOME)
            state["metrics"] = new_metrics()
            state["reset"] = True

        env.on_reset = on_reset
        env.reset()
        state["reset"] = False
        if args.gui:
            env.open_viewer("ex_table_pick_place.py")

        completed = False
        while not completed:
            try:
                completed = all(run_phase(env, robot, fingers, dyn, p, state) for p in phases)
                break
            except ResetRequested:
                continue
        env.update()

        cube = env.body_frame("cube").p
        place_err = math.hypot(cube.x() - PLACE[0], cube.y() - PLACE[1])
        on_table = abs(cube.z() - (SURFACE_Z + CUBE_HS)) < 0.002
        m = state["metrics"]
        ratio = m["peak"] / m["predicted"] if m["predicted"] > 0.0 else 0.0
        elbow_h = m["elbow_min"] - SURFACE_Z
        print(
            f"cube final position: [{cube.x():.4f}, {cube.y():.4f}, {cube.z():.4f}] "
            f"place error {place_err * 1000:.4f} mm (limit {MAX_PLACE_ERR * 1000} mm)"
            + ("" if on_table else ", not on the table")
        )
        print(
            f"push deflection: {m['peak'] * 1000:.4f} mm (at least {MIN_DEFLECTION * 1000} mm), "
            f"stiffness predicts {m['predicted'] * 1000:.4f} mm, ratio {ratio:.4f} "
            f"(limits {DEFLECTION_RATIO[0]}..{DEFLECTION_RATIO[1]})"
        )
        print(
            f"deflection {SETTLE} s after the push: {m['residual'] * 1000:.4f} mm "
            f"(limit {MAX_RESIDUAL * 1000} mm)"
        )
        print(
            f"cube from the TCP while pushed: {m['cube_slip'] * 1000:.4f} mm "
            f"(limit {MAX_CUBE_SLIP * 1000} mm)"
        )
        print(
            f"lowest elbow above the table: {elbow_h * 1000:.4f} mm "
            f"(limit {MIN_ELBOW_HEIGHT * 1000} mm)"
        )
    finally:
        env.close()
    if args.gui:
        return 0
    ok = (
        completed
        and on_table
        and place_err <= MAX_PLACE_ERR
        and m["peak"] >= MIN_DEFLECTION
        and DEFLECTION_RATIO[0] <= ratio <= DEFLECTION_RATIO[1]
        and 0.0 <= m["residual"] <= MAX_RESIDUAL
        and m["cube_slip"] <= MAX_CUBE_SLIP
        and elbow_h >= MIN_ELBOW_HEIGHT
    )
    print(f"{'PASS' if ok else 'FAIL'}: the cube was placed through the push")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
