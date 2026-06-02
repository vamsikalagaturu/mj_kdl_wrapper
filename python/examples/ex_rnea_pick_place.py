#!/usr/bin/env python3
"""RNEA computed-torque pick-place example using PyKDL directly."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import PyKDL as kdl
import mj_kdl_wrapper as mjk

ARM = "third_party/menagerie/kinova_gen3/gen3.xml"
GRIPPER = "third_party/menagerie/robotiq_2f85/2f85.xml"
TABLE = "src/examples/assets/table.xml"
HOME = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
KP = [100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0]
KD = [20.0, 28.0, 20.0, 28.0, 20.0, 28.0, 20.0]
SURFACE_Z = 0.70


def path(value: str) -> Path:
    p = Path(value)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist; run from the mj_kdl_wrapper root")
    return p


def build_scene() -> tuple[mjk.Scene, mjk.Robot]:
    table = mjk.SceneObject()
    table.name = "table"
    table.mjcf_path = str(path(os.environ.get("MJ_KDL_TABLE", TABLE)))
    table.pos = [0.0, 0.0, SURFACE_Z]
    table.fixed = True
    cube = mjk.SceneObject()
    cube.name = "cube"
    cube.shape = mjk.Shape.BOX
    cube.size = [0.02, 0.02, 0.02]
    cube.pos = [0.40, 0.0, SURFACE_Z + 0.02]
    cube.rgba = [0.1, 0.35, 1.0, 1.0]
    cube.mass = 0.1
    cube.friction = [1.0, 0.005, 0.0001]

    attach = mjk.AttachmentSpec()
    attach.mjcf_path = str(path(os.environ.get("MJ_KDL_GRIPPER", GRIPPER)))
    attach.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
    attach.prefix = "g_"
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    spec.objects = [table, cube]
    robot_spec = mjk.RobotSpec()
    robot_spec.path = str(path(os.environ.get("MJ_KDL_MODEL", ARM)))
    robot_spec.pos = [0.0, 0.0, SURFACE_Z]
    robot_spec.attachments = [attach]
    spec.robots = [robot_spec]
    scene = mjk.Scene.build(spec)
    tool = mjk.ToolFrameSpec()
    tool.tool_body = "g_base"
    tool.tcp_site = "g_pinch"
    robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link", tool=tool)
    return scene, robot


def jnt(values: list[float]) -> kdl.JntArray:
    q = kdl.JntArray(len(values))
    for i, v in enumerate(values):
        q[i] = v
    return q


def as_list(q: kdl.JntArray) -> list[float]:
    return [q[i] for i in range(q.rows())]


def solve_position_ik(
    chain: kdl.Chain, seed_values: list[float], target: kdl.Vector
) -> list[float]:
    fk = kdl.ChainFkSolverPos_recursive(chain)
    ik = kdl.ChainIkSolverVel_wdls(chain)
    ik.setLambda(0.05)
    q = jnt(seed_values)
    dq = kdl.JntArray(q.rows())
    for _ in range(700):
        current = kdl.Frame()
        fk.JntToCart(q, current)
        dx = kdl.diff(current, kdl.Frame(current.M, target))
        dx.rot = kdl.Vector.Zero()
        if dx.vel.Norm() < 0.003:
            return as_list(q)
        if dx.vel.Norm() > 0.05:
            dx.vel = dx.vel * (0.05 / dx.vel.Norm())
        if ik.CartToJnt(q, dx, dq) < 0:
            raise RuntimeError("PyKDL IK velocity step failed")
        for i in range(q.rows()):
            q[i] += dq[i]
    raise RuntimeError("PyKDL IK did not converge")


def waypoints(chain: kdl.Chain) -> dict[str, list[float]]:
    seed = HOME[:]

    def solve(pos):
        nonlocal seed
        seed = solve_position_ik(chain, seed, kdl.Vector(*pos))
        return seed[:]

    return {
        "home": HOME[:],
        "pick_above": solve([0.40, 0.0, 0.24]),
        "pick": solve([0.40, 0.0, 0.04]),
        "lift": solve([0.40, 0.0, 0.34]),
        "place_above": solve([0.40, 0.24, 0.24]),
        "place": solve([0.40, 0.24, 0.04]),
    }


def rnea_controller(
    robot: mjk.Robot, solver: kdl.ChainIdSolver_RNE, chain: kdl.Chain, target: list[float]
) -> None:
    robot.update()
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


def run_phase(scene, robot, solver, chain, phase) -> bool:
    print(f"State: {phase['name']}")
    start = robot.jnt_pos_msr[:]
    t0 = scene.time()
    while scene.time() - t0 < phase["duration"]:
        a = max(0.0, min(1.0, (scene.time() - t0) / phase["duration"]))
        target = [x + a * (y - x) for x, y in zip(start, phase["target"])]
        rnea_controller(robot, solver, chain, target)
        if scene.has_actuator("g_fingers_actuator"):
            scene.set_actuator_ctrl("g_fingers_actuator", phase["gripper"])
        if not robot.step():
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    scene, robot = build_scene()
    try:
        chain = robot.kdl_chain()
        solver = kdl.ChainIdSolver_RNE(chain, kdl.Vector(0.0, 0.0, -9.81))
        robot.ctrl_mode = mjk.CtrlMode.TORQUE
        q = waypoints(chain)
        phases = [
            {"name": "HOME", "target": q["home"], "duration": 0.8, "gripper": 0.0},
            {"name": "PICK_ABOVE", "target": q["pick_above"], "duration": 2.0, "gripper": 0.0},
            {"name": "PICK", "target": q["pick"], "duration": 2.0, "gripper": 0.0},
            {"name": "CLOSE", "target": q["pick"], "duration": 1.0, "gripper": 255.0},
            {"name": "LIFT", "target": q["lift"], "duration": 1.5, "gripper": 255.0},
            {"name": "PLACE_ABOVE", "target": q["place_above"], "duration": 1.5, "gripper": 255.0},
            {"name": "PLACE", "target": q["place"], "duration": 2.0, "gripper": 255.0},
            {"name": "OPEN", "target": q["place"], "duration": 0.8, "gripper": 0.0},
            {"name": "RETREAT", "target": q["place_above"], "duration": 1.2, "gripper": 0.0},
        ]
        if args.gui:
            viewer = mjk.SimulateViewer.open(robot, "ex_rnea_pick_place.py")
            try:
                for phase in phases:
                    if not viewer.is_running() or not run_phase(scene, robot, solver, chain, phase):
                        break
            finally:
                viewer.close()
        else:
            for phase in phases:
                if not run_phase(scene, robot, solver, chain, phase):
                    break
        cube_frame = scene.body_frame("cube")
        cube_pos = [cube_frame.p.x(), cube_frame.p.y(), cube_frame.p.z()]
        print(f"cube final position: {[round(x, 3) for x in cube_pos]}")
    finally:
        scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
