#!/usr/bin/env python3
"""RNEA computed-torque pick-place example using PyKDL directly.

Ported from src/examples/ex_rnea_pick_place.cpp. Full computed-torque control
via KDL ChainIdSolver_RNE. An Env on_reset hook re-homes the arm, re-poses the
cube, and opens the gripper so the simulate-UI reset replays the task.
"""

from __future__ import annotations

import argparse
import math

import PyKDL as kdl
import mj_kdl_wrapper as mjk

HOME = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
KP = [100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0]
KD = [20.0, 28.0, 20.0, 28.0, 20.0, 28.0, 20.0]
SURFACE_Z = 0.70
CUBE_HS = 0.02
PICK = [0.40, 0.00]
PLACE = [0.40, 0.24]
CUBE_START = [PICK[0], PICK[1], SURFACE_Z + CUBE_HS]
IK_TOL = 2e-3


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
    cube.friction = [1.0, 0.005, 0.0001]

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
    tool.tool_body = "g_base_mount"
    tool.tcp_site = "g_pinch"
    robot = env.create_robot("base_link", "bracelet_link", tool=tool)
    return env, robot


def jnt(values: list[float]) -> kdl.JntArray:
    q = kdl.JntArray(len(values))
    for i, v in enumerate(values):
        q[i] = v
    return q


def as_list(q: kdl.JntArray) -> list[float]:
    return [q[i] for i in range(q.rows())]


def solve_pose_ik(chain, limits, seed_values: list[float], target: kdl.Frame) -> list[float]:
    fk = kdl.ChainFkSolverPos_recursive(chain)
    ik = kdl.ChainIkSolverVel_wdls(chain)
    ik.setLambda(0.05)
    q = jnt(seed_values)
    dq = kdl.JntArray(q.rows())
    for _ in range(300):
        current = kdl.Frame()
        fk.JntToCart(q, current)
        dx = kdl.diff(current, target)
        if dx.vel.Norm() <= IK_TOL and dx.rot.Norm() <= 2e-2:
            return as_list(q)
        if dx.vel.Norm() > 0.05:
            dx.vel = dx.vel * (0.05 / dx.vel.Norm())
        if dx.rot.Norm() > 0.20:
            dx.rot = dx.rot * (0.20 / dx.rot.Norm())
        if ik.CartToJnt(q, dx, dq) < 0:
            raise RuntimeError("PyKDL IK velocity step failed")
        for i in range(q.rows()):
            lo, hi = limits[i]
            q[i] = min(hi, max(lo, q[i] + dq[i]))
    raise RuntimeError("PyKDL IK did not converge")


def waypoints(robot) -> dict[str, list[float]]:
    chain = robot.kdl_chain()
    limits = [
        (lo, hi) if math.isfinite(lo) and math.isfinite(hi) else (-2 * math.pi, 2 * math.pi)
        for lo, hi in robot.joint_limits
    ]
    grasp_rot = robot.tip_to_tcp.M
    seed = HOME[:]

    # World targets, in the arm base frame (the arm stands on the table top).
    def solve(xy, z):
        nonlocal seed
        target = kdl.Frame(grasp_rot, kdl.Vector(xy[0], xy[1], z))
        seed = solve_pose_ik(chain, limits, seed, target)
        return seed[:]

    z_grasp = CUBE_HS
    return {
        "home": HOME[:],
        "pick_above": solve(PICK, z_grasp + 0.20),
        "pick": solve(PICK, z_grasp),
        "lift": solve(PICK, z_grasp + 0.30),
        "place_above": solve(PLACE, z_grasp + 0.20),
        "place": solve(PLACE, z_grasp),
    }


def rnea_controller(robot, solver, chain, target: list[float]) -> None:
    q = jnt(robot.jnt_pos_msr)
    qdot = jnt(robot.jnt_vel_msr)
    qddot = kdl.JntArray(robot.n_joints)
    tau = kdl.JntArray(robot.n_joints)
    for i in range(robot.n_joints):
        qddot[i] = KP[i] * (target[i] - q[i]) - KD[i] * qdot[i]
    wrenches = [kdl.Wrench.Zero() for _ in range(chain.getNrOfSegments())]
    if solver.CartToJnt(q, qdot, qddot, wrenches, tau) < 0:
        raise RuntimeError("PyKDL RNEA failed")
    robot.jnt_trq_cmd = as_list(tau)


# on_reset flags a UI reset (env is already reset); restart the phases.
def step_once(env, state) -> bool:
    if not env.step():
        return False
    if state["reset"]:
        state["reset"] = False
        raise ResetRequested()
    return True


def run_phase(env, robot, solver, chain, phase, gui, state) -> bool:
    print(f"State: {phase['name']}")
    start = robot.jnt_pos_msr[:]
    t0 = env.time()
    while True:
        t_rel = env.time() - t0
        a = max(0.0, min(1.0, t_rel / phase["duration"]))
        target = [x + a * (y - x) for x, y in zip(start, phase["target"])]
        rnea_controller(robot, solver, chain, target)
        if env.has_actuator("g_fingers_actuator"):
            env.set_actuator_ctrl("g_fingers_actuator", phase["gripper"])
        env.update()

        # Ramp for the duration, then settle to the tolerance, never past the timeout.
        err = max(abs(q - t) for q, t in zip(robot.jnt_pos_msr, phase["target"]))
        settled = phase["tol"] < 0.0 or err <= phase["tol"]
        if (t_rel >= phase["duration"] and settled) or t_rel >= phase["timeout"]:
            return True
        if gui and not env.viewer.is_running():
            return False
        if not step_once(env, state):
            return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    env, robot = build_env()
    try:
        chain = robot.kdl_chain()
        solver = kdl.ChainIdSolver_RNE(chain, kdl.Vector(0.0, 0.0, -9.81))
        robot.set_control_mode(mjk.CtrlMode.TORQUE)
        state = {"reset": False}

        def on_reset(ctx):
            robot.set_joint_pos(HOME)
            env.set_body_pose("cube", CUBE_START)
            if env.has_actuator("g_fingers_actuator"):
                env.set_actuator_ctrl("g_fingers_actuator", 0.0)
            state["reset"] = True

        env.on_reset = on_reset
        env.reset()
        state["reset"] = False

        q = waypoints(robot)

        # name, target, ramp duration [s], timeout [s], settle tolerance [rad] (-1 none), gripper
        def phase(name, target, duration, timeout, tol, gripper):
            return {
                "name": name,
                "target": q[target],
                "duration": duration,
                "timeout": timeout,
                "tol": tol,
                "gripper": gripper,
            }

        phases = [
            phase("HOME", "home", 1.0, 2.5, 0.08, 0.0),
            phase("PICK_ABOVE", "pick_above", 5.0, 7.0, 0.08, 0.0),
            phase("PICK", "pick", 5.0, 8.0, 0.03, 0.0),
            phase("CLOSE", "pick", 1.5, 2.5, -1.0, 0.8),
            phase("LIFT", "lift", 3.0, 5.0, 0.08, 0.8),
            phase("PLACE_ABOVE", "place_above", 3.0, 5.0, 0.08, 0.8),
            phase("PLACE", "place", 5.0, 8.0, 0.03, 0.8),
            phase("OPEN", "place", 1.0, 2.0, -1.0, 0.0),
            phase("RETREAT", "place_above", 2.0, 4.0, 0.08, 0.0),
        ]
        if args.gui:
            env.open_viewer("ex_rnea_pick_place.py")
            while env.viewer.is_running():
                try:
                    for phase in phases:
                        if not run_phase(env, robot, solver, chain, phase, True, state):
                            raise StopIteration
                    break
                except ResetRequested:
                    continue
                except StopIteration:
                    break
        else:
            for phase in phases:
                if not run_phase(env, robot, solver, chain, phase, False, state):
                    break
        cube_frame = env.body_frame("cube")
        cube_pos = [cube_frame.p.x(), cube_frame.p.y(), cube_frame.p.z()]
        place_err_xy = math.hypot(cube_pos[0] - PLACE[0], cube_pos[1] - PLACE[1])
        print(
            f"cube final position: {[round(x, 3) for x in cube_pos]} "
            f"target={PLACE + [SURFACE_Z + CUBE_HS]} xy_error={place_err_xy:.3f}"
        )
    finally:
        env.close()
    return 1 if not args.gui and place_err_xy > 0.08 else 0


if __name__ == "__main__":
    raise SystemExit(main())
