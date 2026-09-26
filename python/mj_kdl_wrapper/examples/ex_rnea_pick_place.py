#!/usr/bin/env python3
"""Two arms facing each other across a table each pick and place a cube with RNEA computed
torque; Python counterpart of ex_rnea_pick_place.cpp. Headless it exits 1 unless both cubes
are placed, both elbows stay up, the free objects stay put, the arms never touch and RNEA
never fails."""

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
# Acceleration gains: the closed loop is qddot = Kp e - Kd qdot [1/s^2], [1/s].
KP = [100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0]
KD = [20.0, 28.0, 20.0, 28.0, 20.0, 28.0, 20.0]
GRIPPER_CLOSED = 0.82  # [rad] top of the 2F-85's ctrlrange (its driver joint stops at 0.8)
MAX_PLACE_ERR = 0.005  # [m] in the table plane
MIN_ELBOW_HEIGHT = 0.45  # [m] forearm_link origin above the table
MAX_DISTURB = 0.001  # [m] free-object displacement

# Base x, y [m] and yaw [rad] on the tabletop: facing each other, each arm in its own half.
BASES = [(-0.70, -0.12, 0.0), (0.70, 0.12, math.pi)]
PREFIXES = ["", "r2_"]
FREE_OBJECTS = [
    ("red_box", mjk.Shape.BOX, 0.0, 0.42, 0.03, (1.0, 0.2, 0.2)),
    ("green_box", mjk.Shape.BOX, 0.0, -0.42, 0.03, (0.2, 1.0, 0.2)),
    ("yellow_box", mjk.Shape.BOX, -0.45, 0.42, 0.04, (1.0, 0.85, 0.1)),
    ("orange_sphere", mjk.Shape.SPHERE, 0.45, -0.42, 0.035, (1.0, 0.55, 0.0)),
    ("purple_sphere", mjk.Shape.SPHERE, 0.0, 0.0, 0.025, (0.7, 0.0, 0.9)),
]
# Both look at the table centre: overview from -y above, side from +y.
CAMERAS = [
    ("overview", [0.0, -1.3, 1.9], [0.410747, 0.0, 0.0, 0.911749], 55.0),
    ("side", [0.0, 1.5, 1.15], [0.0, 0.633989, 0.773342, 0.0], 50.0),
]


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


def world_T_base(arm: int) -> kdl.Frame:
    x, y, yaw = BASES[arm]
    return kdl.Frame(kdl.Rotation.RotZ(yaw), kdl.Vector(x, y, SURFACE_Z))


def cube_spot(arm: int, xy) -> kdl.Vector:
    """A cube's centre resting on the table at xy in the arm's base frame, in the world frame."""
    return world_T_base(arm) * kdl.Vector(xy[0], xy[1], CUBE_HS)


def cube_object(name: str, spot: kdl.Vector) -> mjk.SceneObject:
    cube = mjk.SceneObject()
    cube.name = name
    cube.shape = mjk.Shape.BOX
    cube.size = [CUBE_HS, CUBE_HS, CUBE_HS]
    cube.pos = [spot.x(), spot.y(), spot.z()]
    cube.rgba = [0.1, 0.35, 1.0, 1.0]
    cube.mass = 0.1
    cube.condim = mjk.Condim.Torsional
    cube.friction = [0.8, 0.02, 0.001]
    return cube


def free_object(name, shape, x, y, half, rgb) -> mjk.SceneObject:
    obj = mjk.SceneObject()
    obj.name = name
    obj.shape = shape
    box = shape == mjk.Shape.BOX
    obj.size = [half, half if box else 0.0, half if box else 0.0]
    obj.pos = [x, y, SURFACE_Z + half]
    obj.rgba = [*rgb, 1.0]
    obj.mass = 0.1
    obj.friction = [1.0, 0.005, 0.0001]  # MuJoCo's geom default
    return obj


def camera(name, pos, quat, fovy) -> mjk.CameraSpec:
    cam = mjk.CameraSpec()
    cam.name = name
    cam.pos = pos
    cam.quat = quat
    cam.fovy = fovy
    return cam


def build_env() -> tuple[mjk.Env, list[mjk.Robot]]:
    table = mjk.SceneObject()
    table.name = "table"
    table.mjcf_path = mjk.menagerie.asset_path("table.xml", env_var="MJ_KDL_TABLE")
    table.pos = [0.0, 0.0, SURFACE_Z]
    table.fixed = True
    gripper = mjk.AttachmentSpec()
    gripper.mjcf_path = mjk.menagerie.asset_path("robotiq_2f85/2f85.xml", env_var="MJ_KDL_GRIPPER")
    gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
    gripper.prefix = "g_"
    robot_specs = []
    for (x, y, yaw), prefix in zip(BASES, PREFIXES):
        robot_spec = mjk.RobotSpec()
        robot_spec.path = mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")
        robot_spec.prefix = prefix
        robot_spec.pos = [x, y, SURFACE_Z]
        robot_spec.quat = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
        robot_spec.attachments = [gripper]
        robot_specs.append(robot_spec)

    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    spec.robots = robot_specs
    spec.objects = [
        table,
        *(cube_object(p + "cube", cube_spot(a, PICK)) for a, p in enumerate(PREFIXES)),
        *(free_object(*o) for o in FREE_OBJECTS),
    ]
    spec.cameras = [camera(*c) for c in CAMERAS]
    env = mjk.Env.build(spec)
    tool = mjk.ToolFrameSpec()
    tool.tool_body = "g_base_mount"
    tool.tcp_site = "g_pinch"
    robots = [env.create_robot("base_link", "bracelet_link", p, tool=tool) for p in PREFIXES]
    return env, robots


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
    """Joint waypoints in the arm's base frame, so they serve every arm placed like this one."""
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


def measure(env, state: dict) -> None:
    """What a viewer sees: the elbows, the free objects and any arm-to-arm contact."""
    m = state["metrics"]
    for a, prefix in enumerate(PREFIXES):
        z = env.body_frame(prefix + "forearm_link").p.z()
        m["elbow_min"][a] = min(m["elbow_min"][a], z)
    for name, rest in state["rest"].items():
        m["disturb"] = max(m["disturb"], (env.body_frame(name).p - rest).Norm())
    model, data, roots = env.model, env.data, state["roots"]
    for c in range(data.ncon):
        r0, r1 = (model.body_rootid[model.geom_bodyid[g]] for g in data.contact[c].geom)
        if {r0, r1} == roots:
            m["arm_contacts"] += 1


def run_phase(env, robots, fingers, solvers, phase: Phase, state: dict) -> bool:
    """One update() per step for both arms: it reads the state and applies the previous
    cycle's command."""
    print(f"State: {phase.name}")
    starts = [r.jnt_pos_msr[:] for r in robots]
    t0 = env.data.time
    while True:
        env.update()
        elapsed = env.data.time - t0
        alpha = min(1.0, max(0.0, elapsed / phase.duration)) if phase.duration > 0.0 else 1.0
        err = 0.0
        for robot, finger, (rnea, n_seg), start in zip(robots, fingers, solvers, starts):
            q_des = [s + alpha * (t - s) for s, t in zip(start, phase.target)]
            robot.jnt_trq_cmd = rnea_torque(robot, rnea, n_seg, q_des)
            finger.ctrl[0] = phase.gripper
            err = max(err, *(abs(t - m) for t, m in zip(phase.target, robot.jnt_pos_msr)))
        done_pose = phase.settle_tol < 0.0 or err <= phase.settle_tol
        if (elapsed >= phase.duration and done_pose) or elapsed >= phase.timeout:
            return True
        if not env.step():
            return False
        if state["reset"]:
            state["reset"] = False
            raise ResetRequested()
        env.pace()
        measure(env, state)


def new_metrics() -> dict:
    return {"elbow_min": [math.inf, math.inf], "disturb": 0.0, "arm_contacts": 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    env, robots = build_env()
    try:
        print(f"cameras: {' '.join(env.model.camera(i).name for i in range(env.model.ncam))}")
        fingers = [env.data.actuator(p + "g_fingers_actuator") for p in PREFIXES]
        gravity_vec = kdl.Vector(0.0, 0.0, env.spec.gravity_z)
        solvers, dyns = [], []
        for robot in robots:
            robot.set_control_mode(mjk.CtrlMode.TORQUE)
            chain = robot.kdl_chain()
            solvers.append((kdl.ChainIdSolver_RNE(chain, gravity_vec), chain.getNrOfSegments()))
            dyns.append(kdl.ChainDynParam(chain, gravity_vec))
        # Same waypoints for both arms: each target is in its own base frame.
        phases = pick_place_phases(robots[0])
        state = {
            "reset": False,
            "metrics": new_metrics(),
            "rest": {name: kdl.Vector(x, y, SURFACE_Z + h) for name, _, x, y, h, _ in FREE_OBJECTS},
            "roots": {env.model.body_rootid[env.model.body(p + "base_link").id] for p in PREFIXES},
        }

        def on_reset(ctx):
            for a, (robot, finger, dyn) in enumerate(zip(robots, fingers, dyns)):
                robot.set_joint_pos(HOME)
                spot = cube_spot(a, PICK)
                env.set_body_pose(PREFIXES[a] + "cube", [spot.x(), spot.y(), spot.z()])
                finger.ctrl[0] = 0.0
                robot.jnt_trq_cmd = gravity(dyn, HOME)
            state["metrics"] = new_metrics()
            state["reset"] = True

        env.on_reset = on_reset
        env.reset()
        state["reset"] = False
        if args.gui:
            env.open_viewer("ex_rnea_pick_place.py")

        completed = False
        while not completed:
            try:
                completed = all(run_phase(env, robots, fingers, solvers, p, state) for p in phases)
                break
            except ResetRequested:
                continue
        env.update()

        m = state["metrics"]
        ok = completed
        for a, prefix in enumerate(PREFIXES):
            cube = env.body_frame(prefix + "cube").p
            spot = cube_spot(a, PLACE)
            place_err = math.hypot(cube.x() - spot.x(), cube.y() - spot.y())
            on_table = abs(cube.z() - spot.z()) < 0.002
            elbow_h = m["elbow_min"][a] - SURFACE_Z
            print(
                f"arm {a + 1}: cube at [{cube.x():.4f}, {cube.y():.4f}, {cube.z():.4f}] "
                f"place error {place_err * 1000:.4f} mm (limit {MAX_PLACE_ERR * 1000} mm)"
                + ("" if on_table else ", not on the table")
                + f"; lowest elbow above the table {elbow_h * 1000:.4f} mm "
                f"(limit {MIN_ELBOW_HEIGHT * 1000} mm)"
            )
            ok = ok and on_table and place_err <= MAX_PLACE_ERR and elbow_h >= MIN_ELBOW_HEIGHT
        print(
            f"largest free-object displacement: {m['disturb'] * 1000:.4f} mm "
            f"(limit {MAX_DISTURB * 1000} mm)"
        )
        print(f"arm-to-arm contacts: {m['arm_contacts']} (limit 0)")
        ok = ok and m["disturb"] <= MAX_DISTURB and m["arm_contacts"] == 0
    finally:
        env.close()
    if args.gui:
        return 0
    print(f"{'PASS' if ok else 'FAIL'}: both arms placed their cubes and left the rest alone")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
