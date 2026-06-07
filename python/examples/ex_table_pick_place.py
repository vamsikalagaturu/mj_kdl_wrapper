#!/usr/bin/env python3
"""Table pick-place example ported from src/examples/ex_table_pick_place.cpp.

Picks a cube from one table location and places it at another using joint
impedance control. An Env on_reset hook re-homes the arm, re-poses the cube,
and opens the gripper so the simulate-UI reset replays the task.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import PyKDL as kdl
import mj_kdl_wrapper as mjk

ARM = "third_party/menagerie/kinova_gen3/gen3.xml"
GRIPPER = "third_party/menagerie/robotiq_2f85/2f85.xml"
TABLE = "src/examples/assets/table.xml"
HOME = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
SURFACE_Z = 0.70
CUBE_HALF_SIZE = 0.02
PICK_X = 0.40
PICK_Y = 0.00
PLACE_X = 0.40
PLACE_Y = 0.24
KP = [100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0]
KD = [10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0]
CUBE_START = [PICK_X, PICK_Y, SURFACE_Z + CUBE_HALF_SIZE]


class ResetRequested(Exception):
    """Raised when the simulate UI reset is detected, to restart the sequence."""


@dataclass(frozen=True)
class Phase:
    name: str
    target: list[float]
    duration: float
    timeout: float
    settle_tol: float
    gripper: float


def path(value: str, label: str) -> Path:
    p = Path(value)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist for {label}; run from the repo root")
    return p


def gripper_attachment(gripper_path: Path) -> mjk.AttachmentSpec:
    attach = mjk.AttachmentSpec()
    attach.mjcf_path = str(gripper_path)
    attach.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
    attach.prefix = "g_"
    return attach


def table_object(table_path: Path) -> mjk.SceneObject:
    table = mjk.SceneObject()
    table.name = "table"
    table.mjcf_path = str(table_path)
    table.pos = [0.0, 0.0, SURFACE_Z]
    table.fixed = True
    return table


def cube_object() -> mjk.SceneObject:
    cube = mjk.SceneObject()
    cube.name = "cube"
    cube.shape = mjk.Shape.BOX
    cube.size = [CUBE_HALF_SIZE, CUBE_HALF_SIZE, CUBE_HALF_SIZE]
    cube.pos = CUBE_START[:]
    cube.rgba = [0.1, 0.35, 1.0, 1.0]
    cube.mass = 0.1
    cube.condim = mjk.Condim.Torsional
    cube.friction = [0.8, 0.02, 0.001]
    return cube


def build_env() -> tuple[mjk.Env, mjk.Robot]:
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    spec.objects = [
        table_object(path(os.environ.get("MJ_KDL_TABLE", TABLE), "table model")),
        cube_object(),
    ]

    robot_spec = mjk.RobotSpec()
    robot_spec.path = str(path(mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL"), "arm model"))
    robot_spec.pos = [0.0, 0.0, SURFACE_Z]
    robot_spec.attachments = [
        gripper_attachment(path(mjk.menagerie.model_path("robotiq_2f85", env_var="MJ_KDL_GRIPPER"), "gripper model"))
    ]
    spec.robots = [robot_spec]

    env = mjk.Env.build(spec)
    tool = mjk.ToolFrameSpec()
    tool.tool_body = "g_base"
    tool.tcp_site = "g_pinch"
    robot = env.create_robot("base_link", "bracelet_link", tool=tool)
    return env, robot


def jnt(values: list[float]) -> kdl.JntArray:
    q = kdl.JntArray(len(values))
    for i, value in enumerate(values):
        q[i] = value
    return q


def as_list(q: kdl.JntArray) -> list[float]:
    return [q[i] for i in range(q.rows())]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_joint(value: float, limit: tuple[float, float]) -> float:
    low, high = limit
    if math.isfinite(low) and math.isfinite(high) and high > low:
        return clamp(value, low, high)
    return value


def solve_position_ik(
    chain: kdl.Chain,
    seed_values: list[float],
    target: kdl.Frame,
    joint_limits: list[tuple[float, float]],
) -> list[float]:
    fk = kdl.ChainFkSolverPos_recursive(chain)
    ik = kdl.ChainIkSolverVel_wdls(chain)
    ik.setLambda(0.05)
    q = jnt(seed_values)
    dq = kdl.JntArray(q.rows())
    for _ in range(700):
        current = kdl.Frame()
        fk.JntToCart(q, current)
        dx = kdl.diff(current, target)
        pos_err = dx.vel.Norm()
        rot_err = dx.rot.Norm()
        if pos_err <= 0.003 and rot_err <= 0.03:
            return as_list(q)
        if pos_err > 0.05:
            dx.vel = dx.vel * (0.05 / pos_err)
        if rot_err > 0.20:
            dx.rot = dx.rot * (0.20 / rot_err)
        if ik.CartToJnt(q, dx, dq) < 0:
            raise RuntimeError("PyKDL IK velocity step failed")
        for i in range(q.rows()):
            q[i] = clamp_joint(q[i] + dq[i], joint_limits[i])
    raise RuntimeError("PyKDL IK did not converge")


def build_waypoints(env: mjk.Env, robot: mjk.Robot) -> dict[str, list[float]]:
    chain = robot.kdl_chain()
    target_rot = (env.body_frame("bracelet_link").Inverse() * env.site_frame("g_pinch")).M
    seed = HOME[:]

    def solve(x: float, y: float, z: float) -> list[float]:
        nonlocal seed
        seed = solve_position_ik(
            chain, seed, kdl.Frame(target_rot, kdl.Vector(x, y, z)), robot.joint_limits
        )
        return seed[:]

    z_grasp = CUBE_HALF_SIZE
    z_above = z_grasp + 0.20
    z_lift = z_grasp + 0.30
    return {
        "home": HOME[:],
        "pick_above": solve(PICK_X, PICK_Y, z_above),
        "pick": solve(PICK_X, PICK_Y, z_grasp),
        "lift": solve(PICK_X, PICK_Y, z_lift),
        "place_above": solve(PLACE_X, PLACE_Y, z_above),
        "place": solve(PLACE_X, PLACE_Y, z_grasp),
    }


def apply_pd_gravity(robot: mjk.Robot, target: list[float]) -> None:
    robot.update()
    gravity = robot.gravity_torques(-9.81)
    robot.jnt_trq_cmd = [
        KP[i] * (target[i] - robot.jnt_pos_msr[i]) - KD[i] * robot.jnt_vel_msr[i] + gravity[i]
        for i in range(robot.n_joints)
    ]


def max_abs_joint_err(robot: mjk.Robot, target: list[float]) -> float:
    return max(abs(target[i] - robot.jnt_pos_msr[i]) for i in range(robot.n_joints))


def lerp(start: list[float], target: list[float], alpha: float) -> list[float]:
    return [a + alpha * (b - a) for a, b in zip(start, target)]


def step_once(
    env: mjk.Env, robot: mjk.Robot, viewer: mjk.SimulateViewer | None, state: dict
) -> bool:
    ok = viewer.step() if viewer is not None else robot.step()
    if not ok:
        return False
    if viewer is not None and env.time() < state["prev"] - 1e-6:
        env.reset()
        state["prev"] = env.time()
        raise ResetRequested()
    state["prev"] = env.time()
    return True


def run_phase(
    env: mjk.Env,
    robot: mjk.Robot,
    phase: Phase,
    viewer: mjk.SimulateViewer | None,
    state: dict,
) -> bool:
    print(f"State: {phase.name}")
    robot.update()
    start = robot.jnt_pos_msr[:]
    t0 = env.time()
    while True:
        elapsed = env.time() - t0
        alpha = clamp(elapsed / phase.duration, 0.0, 1.0) if phase.duration > 0.0 else 1.0
        apply_pd_gravity(robot, lerp(start, phase.target, alpha))
        if env.has_actuator("g_fingers_actuator"):
            env.set_actuator_ctrl("g_fingers_actuator", phase.gripper)

        done_time = elapsed >= phase.duration
        done_pose = phase.settle_tol < 0.0 or max_abs_joint_err(robot, phase.target) <= phase.settle_tol
        done_timeout = phase.timeout > 0.0 and elapsed >= phase.timeout
        if (done_time and done_pose) or done_timeout:
            return True
        if viewer is not None and not viewer.is_running():
            return False
        if not step_once(env, robot, viewer, state):
            return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    env, robot = build_env()
    try:
        robot.ctrl_mode = mjk.CtrlMode.TORQUE

        def on_reset(ctx):
            robot.set_joint_pos(HOME, call_forward=False)
            env.set_body_pose("cube", CUBE_START)
            if env.has_actuator("g_fingers_actuator"):
                env.set_actuator_ctrl("g_fingers_actuator", 0.0)

        env.on_reset = on_reset
        env.reset()

        waypoints = build_waypoints(env, robot)
        phases = [
            Phase("HOME", waypoints["home"], 1.0, 2.5, 0.08, 0.0),
            Phase("PICK_ABOVE", waypoints["pick_above"], 5.0, 7.0, 0.08, 0.0),
            Phase("PICK", waypoints["pick"], 5.0, 8.0, 0.03, 0.0),
            Phase("CLOSE", waypoints["pick"], 1.5, 2.5, -1.0, 255.0),
            Phase("LIFT", waypoints["lift"], 3.0, 5.0, 0.08, 255.0),
            Phase("PLACE_ABOVE", waypoints["place_above"], 3.0, 5.0, 0.08, 255.0),
            Phase("PLACE", waypoints["place"], 5.0, 8.0, 0.03, 255.0),
            Phase("OPEN", waypoints["place"], 1.0, 2.0, -1.0, 0.0),
            Phase("RETREAT", waypoints["place_above"], 2.0, 4.0, 0.08, 0.0),
            Phase("HOLD", waypoints["place_above"], 1.0 if not args.gui else 10.0, 0.0, -1.0, 0.0),
        ]
        state = {"prev": env.time()}
        if args.gui:
            viewer = mjk.SimulateViewer.open(robot, "ex_table_pick_place.py")
            try:
                while viewer.is_running():
                    try:
                        for phase in phases:
                            if not run_phase(env, robot, phase, viewer, state):
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
                if not run_phase(env, robot, phase, None, state):
                    break
        cube = env.body_frame("cube")
        cube_pos = [cube.p.x(), cube.p.y(), cube.p.z()]
        place_err = math.hypot(cube_pos[0] - PLACE_X, cube_pos[1] - PLACE_Y)
        print(f"cube final position: {[round(v, 3) for v in cube_pos]} xy_error={place_err:.3f}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
