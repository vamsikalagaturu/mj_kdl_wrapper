#!/usr/bin/env python3
"""Scripted floor pick example ported from src/examples/ex_pick.cpp."""

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
HOME = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
CUBE_X = 0.40
CUBE_Y = 0.00
CUBE_HALF_SIZE = 0.02
CUBE_Z = CUBE_HALF_SIZE
KP = [100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0]
KD = [10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0]


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


def cube_object() -> mjk.SceneObject:
    cube = mjk.SceneObject()
    cube.name = "cube"
    cube.shape = mjk.Shape.BOX
    cube.size = [CUBE_HALF_SIZE, CUBE_HALF_SIZE, CUBE_HALF_SIZE]
    cube.pos = [CUBE_X, CUBE_Y, CUBE_Z]
    cube.rgba = [1.0, 0.5, 0.0, 1.0]
    cube.mass = 0.1
    cube.condim = mjk.Condim.Torsional
    cube.friction = [0.8, 0.02, 0.001]
    return cube


def build_scene() -> tuple[mjk.Scene, mjk.Robot]:
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    spec.objects = [cube_object()]

    robot_spec = mjk.RobotSpec()
    robot_spec.path = str(path(os.environ.get("MJ_KDL_MODEL", ARM), "arm model"))
    robot_spec.attachments = [
        gripper_attachment(path(os.environ.get("MJ_KDL_GRIPPER", GRIPPER), "gripper model"))
    ]
    spec.robots = [robot_spec]

    scene = mjk.Scene.build(spec)
    tool = mjk.ToolFrameSpec()
    tool.tool_body = "g_base"
    tool.tcp_site = "g_pinch"
    robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link", tool=tool)
    return scene, robot


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


def build_waypoints(scene: mjk.Scene, robot: mjk.Robot) -> dict[str, list[float]]:
    chain = robot.kdl_chain()
    target_rot = (scene.body_frame("bracelet_link").Inverse() * scene.site_frame("g_pinch")).M
    seed = HOME[:]

    def solve(x: float, y: float, z: float) -> list[float]:
        nonlocal seed
        seed = solve_position_ik(
            chain, seed, kdl.Frame(target_rot, kdl.Vector(x, y, z)), robot.joint_limits
        )
        return seed[:]

    return {
        "home": HOME[:],
        "pregrasp": solve(CUBE_X, CUBE_Y, CUBE_Z + 0.20),
        "grasp": solve(CUBE_X, CUBE_Y, CUBE_Z),
        "lift": solve(CUBE_X, CUBE_Y, CUBE_Z + 0.30),
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


def step_once(robot: mjk.Robot, viewer: mjk.SimulateViewer | None) -> bool:
    if viewer is not None:
        return viewer.step()
    return robot.step()


def run_phase(
    scene: mjk.Scene,
    robot: mjk.Robot,
    phase: Phase,
    viewer: mjk.SimulateViewer | None,
) -> bool:
    print(f"State: {phase.name}")
    robot.update()
    start = robot.jnt_pos_msr[:]
    t0 = scene.time()
    while True:
        elapsed = scene.time() - t0
        alpha = clamp(elapsed / phase.duration, 0.0, 1.0) if phase.duration > 0.0 else 1.0
        apply_pd_gravity(robot, lerp(start, phase.target, alpha))
        if scene.has_actuator("g_fingers_actuator"):
            scene.set_actuator_ctrl("g_fingers_actuator", phase.gripper)

        done_time = elapsed >= phase.duration
        done_pose = phase.settle_tol < 0.0 or max_abs_joint_err(robot, phase.target) <= phase.settle_tol
        done_timeout = phase.timeout > 0.0 and elapsed >= phase.timeout
        if (done_time and done_pose) or done_timeout:
            return True
        if viewer is not None and not viewer.is_running():
            return False
        if not step_once(robot, viewer):
            return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    scene, robot = build_scene()
    try:
        robot.ctrl_mode = mjk.CtrlMode.TORQUE
        robot.set_joint_pos(HOME, call_forward=True)
        scene.set_body_pose("cube", [CUBE_X, CUBE_Y, CUBE_Z])
        waypoints = build_waypoints(scene, robot)
        phases = [
            Phase("HOME", waypoints["home"], 1.0, 2.5, 0.08, 0.0),
            Phase("PREGRASP", waypoints["pregrasp"], 5.0, 7.0, 0.08, 0.0),
            Phase("GRASP", waypoints["grasp"], 5.0, 8.0, 0.03, 0.0),
            Phase("CLOSE", waypoints["grasp"], 1.5, 2.5, -1.0, 255.0),
            Phase("LIFT", waypoints["lift"], 3.0, 5.0, 0.08, 255.0),
            Phase("HOLD", waypoints["lift"], 1.0 if not args.gui else 10.0, 0.0, -1.0, 255.0),
        ]
        if args.gui:
            viewer = mjk.SimulateViewer.open(robot, "ex_pick.py")
            try:
                for phase in phases:
                    if not run_phase(scene, robot, phase, viewer):
                        break
            finally:
                viewer.close()
        else:
            for phase in phases:
                if not run_phase(scene, robot, phase, None):
                    break
        cube = scene.body_frame("cube")
        print(f"final cube height: {cube.p.z():.3f} m")
    finally:
        scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
