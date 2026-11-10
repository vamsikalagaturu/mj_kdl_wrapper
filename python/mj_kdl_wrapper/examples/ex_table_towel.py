#!/usr/bin/env python3
"""The Gen3 drags a towel on the table with RNEA computed torque; Python counterpart of
ex_table_towel.cpp. Headless it exits 1 unless the towel lay on the table while dragged, came clear
of it at the end, and stayed finite.

Timestep 2 ms, not the 1 ms of the other examples: every step solves the cloth implicitly, and the
compliant torque-controlled arm and the lightly damped towel keep each other swinging, so the
solver iterates each step. At 1 ms that runs at about 0.8x real time on a laptop CPU, at 2 ms about
1.8x; MuJoCo's modeling guide gives 1-4 ms as typical for cloth under discrete."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import PyKDL as kdl

import mj_kdl_wrapper as mjk
from mj_kdl_wrapper.examples.ex_table_pick_place import solve_near_seed

HOME = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
SURFACE_Z = 0.70
KP = [100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0]
KD = [20.0, 28.0, 20.0, 28.0, 20.0, 28.0, 20.0]
GRIPPER_CLOSED = 0.82  # [rad] top of the 2F-85's ctrlrange
TOWEL_LENGTH = 0.30  # [m] the towel hangs this far below the TCP
ON_TABLE = 0.06  # [m] of towel lying on the table while dragged
DRAG_Y = 0.25  # [m] left and right of the start
TOUCH_HEIGHT = 0.010  # [m] above the table: a vertex on it
MIN_ON_TABLE = 9  # towel vertices on the table while dragged
MIN_CLEARANCE = 0.02  # [m] lowest vertex above the table at the end


@dataclass
class Phase:
    name: str
    target: list[float]
    duration: float
    timeout: float
    settle_tol: float
    gripper: float


class ResetRequested(Exception):
    pass


def jnt(values) -> kdl.JntArray:
    q = kdl.JntArray(len(values))
    for i, value in enumerate(values):
        q[i] = value
    return q


def build_env() -> tuple[mjk.Env, mjk.Robot]:
    spec = mjk.SceneSpec()
    spec.timestep = 0.002  # see the module docstring
    spec.use_stage = True
    spec.integrator = mjk.Integrator.Discrete
    spec.solver = mjk.Solver.Newton
    table = mjk.SceneObject()
    table.name = "table"
    table.mjcf_path = str(mjk.ASSETS_DIR / "table.xml")
    table.pos = [0.0, 0.0, SURFACE_Z]
    table.fixed = True
    spec.objects = [table]

    gripper = mjk.AttachmentSpec()
    gripper.mjcf_path = str(mjk.ASSETS_DIR / "robotiq_2f85/2f85.xml")
    gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
    gripper.prefix = "g_"
    towel = mjk.AttachmentSpec()
    towel.mjcf_path = str(mjk.ASSETS_DIR / "towel.xml")
    towel.attach_to = mjk.AttachTarget(mjk.AttachKind.Body, "g_base")
    towel.prefix = "towel_"
    robot_spec = mjk.RobotSpec()
    robot_spec.path = str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")
    robot_spec.pos = [0.0, 0.0, SURFACE_Z]
    robot_spec.attachments = [gripper, towel]
    spec.robots = [robot_spec]

    env = mjk.Env.build(spec)
    env.model.opt.tolerance = 1e-6  # MuJoCo's modeling guide, for cloth under discrete + Newton
    tool = mjk.ToolFrameSpec()
    tool.tool_body = "g_base_mount"
    tool.tcp_site = "g_pinch"
    return env, env.create_robot("base_link", "bracelet_link", tool=tool)


def phases(robot: mjk.Robot) -> list[Phase]:
    """The home TCP pose moved down, left, right and up, the orientation kept; IK once."""
    chain = robot.kdl_chain()
    x_home = kdl.Frame()
    kdl.ChainFkSolverPos_recursive(chain).JntToCart(jnt(HOME), x_home)
    down = (TOWEL_LENGTH - ON_TABLE) - x_home.p.z()
    waypoints, seed = [], HOME
    offsets = ((0.0, 0.0, down), (0.0, DRAG_Y, down), (0.0, -DRAG_Y, down), (0.0, -DRAG_Y, 0.0))
    for offset in offsets:
        target = kdl.Frame(x_home.M, x_home.p + kdl.Vector(*offset))
        seed = solve_near_seed(chain, robot.joint_limits, seed, target)
        waypoints.append(seed)
    low, left, right, up = waypoints
    closed = GRIPPER_CLOSED
    return [
        Phase("CLOSE", HOME, 1.0, 1.0, -1.0, closed),
        Phase("DOWN", low, 2.0, 3.0, 0.03, closed),
        Phase("DRAG_LEFT", left, 2.0, 3.0, 0.03, closed),
        Phase("DRAG_RIGHT", right, 3.0, 4.0, 0.03, closed),
        Phase("UP", up, 2.0, 3.0, 0.03, closed),
        Phase("HOLD", up, 1.0, 1.0, -1.0, closed),
    ]


def rnea_torque(robot, rnea, n_segments, q_des) -> list[float]:
    n = robot.n_joints
    q, qd = robot.jnt_pos_msr, robot.jnt_vel_msr
    qdd = kdl.JntArray(n)
    for i in range(n):
        qdd[i] = KP[i] * (q_des[i] - q[i]) - KD[i] * qd[i]
    tau = kdl.JntArray(n)
    wrenches = [kdl.Wrench.Zero() for _ in range(n_segments)]
    if rnea.CartToJnt(jnt(q), jnt(qd), qdd, wrenches, tau) < 0:
        raise RuntimeError("PyKDL RNEA failed")
    return [tau[i] for i in range(n)]


def towel_heights(env) -> np.ndarray:
    m, d = env.model, env.data
    first, count = m.flex_vertadr[0], m.flex_vertnum[0]
    return d.flexvert_xpos[first : first + count] - [0.0, 0.0, SURFACE_Z]


def run_phase(env, robot, fingers, solver, phase: Phase, state: dict) -> bool:
    """One update() per step: it reads the state and applies the previous cycle's command."""
    print(f"State: {phase.name}")
    start = robot.jnt_pos_msr[:]
    t0 = env.data.time
    while True:
        env.update()
        elapsed = env.data.time - t0
        alpha = min(1.0, max(0.0, elapsed / phase.duration)) if phase.duration > 0.0 else 1.0
        q_des = [s + alpha * (t - s) for s, t in zip(start, phase.target)]
        robot.jnt_trq_cmd = rnea_torque(robot, *solver, q_des)
        fingers.ctrl[0] = phase.gripper
        err = max(abs(t - m) for t, m in zip(phase.target, robot.jnt_pos_msr))
        done_pose = phase.settle_tol < 0.0 or err <= phase.settle_tol
        if (elapsed >= phase.duration and done_pose) or elapsed >= phase.timeout:
            return True
        if not env.step():
            return False
        state["viewer"].pace()
        if state["reset"]:
            state["reset"] = False
            raise ResetRequested()
        if phase.name in ("DRAG_LEFT", "DRAG_RIGHT"):
            on_table = int((towel_heights(env)[:, 2] < TOUCH_HEIGHT).sum())
            state["on_table_max"] = max(state["on_table_max"], on_table)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    env, robot = build_env()
    viewer = mjk.Viewer()
    viewer.paused = True
    try:
        robot.set_control_mode(mjk.CtrlMode.TORQUE)
        chain = robot.kdl_chain()
        gravity_vec = kdl.Vector(0.0, 0.0, env.spec.gravity_z)
        solver = (kdl.ChainIdSolver_RNE(chain, gravity_vec), chain.getNrOfSegments())
        dyn = kdl.ChainDynParam(chain, gravity_vec)
        fingers = env.data.actuator("g_fingers_actuator")
        plan = phases(robot)
        state = {"reset": False, "on_table_max": 0}

        def on_reset(ctx):
            robot.set_joint_pos(HOME)
            fingers.ctrl[0] = 0.0
            g = kdl.JntArray(len(HOME))
            dyn.JntToGravity(jnt(HOME), g)
            robot.jnt_trq_cmd = [g[i] for i in range(g.rows())]
            state["on_table_max"] = 0
            state["reset"] = True

        env.on_reset = on_reset
        env.reset()
        state["reset"] = False
        if args.gui:
            viewer.open(env, "ex_table_towel.py")
        state["viewer"] = viewer

        completed = False
        while not completed:
            try:
                completed = all(run_phase(env, robot, fingers, solver, p, state) for p in plan)
                break
            except ResetRequested:
                continue
        env.update()

        heights = towel_heights(env)
        finite = bool(np.isfinite(heights).all())
        lowest = float(heights[:, 2].min())
        count = len(heights)
        print(
            f"towel vertices on the table while dragged: {state['on_table_max']}/{count} "
            f"(at least {MIN_ON_TABLE})"
        )
        print(
            f"lowest towel vertex at the end: {lowest:.3f} m above the table "
            f"(at least {MIN_CLEARANCE} m)" + ("" if finite else "; the cloth went non-finite")
        )
    finally:
        viewer.close()
        env.close()
    if args.gui:
        return 0
    ok = completed and finite and state["on_table_max"] >= MIN_ON_TABLE and lowest >= MIN_CLEARANCE
    print(f"{'PASS' if ok else 'FAIL'}: the towel was dragged on the table and lifted clear")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
