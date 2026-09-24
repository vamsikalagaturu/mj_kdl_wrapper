#!/usr/bin/env python3
"""Table pour example ported from src/examples/ex_table_pour.cpp."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import PyKDL as kdl
import mj_kdl_wrapper as mjk

HOME = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
TABLE_Z = 0.70
ROBOT_BACK_X = -0.26
JUG_X = 0.30
JUG_Y = 0.14
RETREAT_X = JUG_X - 0.08
RETREAT_Y = JUG_Y - 0.08
JUG_RADIUS = 0.028
JUG_HEIGHT = 0.084
BALL_RADIUS = 0.007
NUM_BALLS = 36
POUR_TILT_RAD = 1.95
TILT_OUTLET_Z = TABLE_Z + 0.18
GRIPPER_CLOSED = 0.8
MIN_BALLS_IN_RECEIVER = 4
IK_TOL = 3e-3
KP = [120.0, 220.0, 120.0, 220.0, 110.0, 190.0, 90.0]
KD = [12.0, 22.0, 12.0, 22.0, 11.0, 18.0, 9.0]


@dataclass(frozen=True)
class Phase:
    name: str
    target: list[float]
    duration: float
    timeout: float
    settle_tol: float
    gripper: float


class ResetRequested(Exception):
    """Raised when the simulate UI reset is detected, to restart the sequence."""


def gripper_attachment(gripper_path: str) -> mjk.AttachmentSpec:
    attach = mjk.AttachmentSpec()
    attach.mjcf_path = gripper_path
    attach.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
    attach.prefix = "g_"
    return attach


def bottle_attachment(bottle_path: str) -> mjk.AttachmentSpec:
    attach = mjk.AttachmentSpec()
    attach.mjcf_path = bottle_path
    attach.attach_to = mjk.AttachTarget(mjk.AttachKind.Body, "g_base")
    attach.prefix = "pour_"
    return attach


def table_object(table_path: str) -> mjk.SceneObject:
    table = mjk.SceneObject()
    table.name = "table"
    table.mjcf_path = table_path
    table.pos = [0.0, 0.0, TABLE_Z]
    table.fixed = True
    return table


def receiver_object(receiver_path: str) -> mjk.SceneObject:
    recv = mjk.SceneObject()
    recv.name = "recv"
    recv.mjcf_path = receiver_path
    recv.pos = [JUG_X, JUG_Y, TABLE_Z]
    return recv


def ball_object(index: int) -> mjk.SceneObject:
    ball = mjk.SceneObject()
    ball.name = f"grain_{index:02d}"
    ball.shape = mjk.Shape.SPHERE
    ball.size = [BALL_RADIUS, 0.0, 0.0]
    ball.pos = [0.0, 0.0, TABLE_Z + 0.40 + index * 2.0 * BALL_RADIUS]
    ball.rgba = [1.0, 0.84, 0.30, 1.0]
    ball.mass = 0.006
    ball.condim = mjk.Condim.Torsional
    ball.friction = [0.5, 0.02, 0.001]
    return ball


def build_env() -> tuple[mjk.Env, mjk.Robot]:
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    spec.objects = [
        table_object(mjk.menagerie.asset_path("table.xml", env_var="MJ_KDL_TABLE")),
        *[ball_object(i) for i in range(NUM_BALLS)],
        receiver_object(mjk.menagerie.asset_path("mug_table.xml", env_var="MJ_KDL_RECEIVER")),
    ]

    robot_spec = mjk.RobotSpec()
    robot_spec.path = mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")
    robot_spec.pos = [ROBOT_BACK_X, 0.0, TABLE_Z]
    robot_spec.attachments = [
        gripper_attachment(
            mjk.menagerie.asset_path("robotiq_2f85/2f85.xml", env_var="MJ_KDL_GRIPPER")
        ),
        bottle_attachment(mjk.menagerie.asset_path("mug.xml", env_var="MJ_KDL_BOTTLE")),
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


def joint_limit_arrays(robot: mjk.Robot) -> tuple[kdl.JntArray, kdl.JntArray]:
    q_min = kdl.JntArray(robot.n_joints)
    q_max = kdl.JntArray(robot.n_joints)
    for i, (low, high) in enumerate(robot.joint_limits):
        if math.isfinite(low) and math.isfinite(high) and high > low:
            q_min[i], q_max[i] = low, high
        else:
            q_min[i], q_max[i] = -2.0 * math.pi, 2.0 * math.pi
    return q_min, q_max


def build_waypoints(env: mjk.Env, robot: mjk.Robot) -> dict[str, list[float]]:
    chain = robot.kdl_chain()
    n = robot.n_joints
    fk = kdl.ChainFkSolverPos_recursive(chain)
    q_min, q_max = joint_limit_arrays(robot)
    ik_vel = kdl.ChainIkSolverVel_pinv(chain)
    ik_nr = kdl.ChainIkSolverPos_NR_JL(chain, q_min, q_max, fk, ik_vel, 2000, 1e-5)
    ik_lma = kdl.ChainIkSolverPos_LMA(chain, 1e-5, 2000)

    q_home = jnt(HOME)
    home_fk = kdl.Frame()
    fk.JntToCart(q_home, home_fk)
    # Carry the bottle at the home orientation, tilted slightly forward.
    carry_tcp = home_fk.M * kdl.Rotation.RotY(-0.05)

    world_T_base = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(ROBOT_BACK_X, 0.0, TABLE_Z))
    base_T_world = world_T_base.Inverse()

    # Constant TCP->outlet offset, measured at the live home configuration.
    robot.set_joint_pos(HOME, call_forward=False)
    world_T_outlet = env.site_frame("pour_outlet")
    world_T_tcp = env.site_frame("g_pinch")
    tcp_outlet = world_T_tcp.Inverse() * world_T_outlet.p

    def outlet_target_to_tcp_target(tcp_rot: kdl.Rotation, outlet_pos: kdl.Vector) -> kdl.Frame:
        return kdl.Frame(tcp_rot, outlet_pos - tcp_rot * tcp_outlet)

    def solve(name: str, seed_values: list[float], outlet_pos: kdl.Vector) -> list[float]:
        target = base_T_world * outlet_target_to_tcp_target(carry_tcp, outlet_pos)
        seed = jnt(seed_values)
        out = kdl.JntArray(n)
        ok = ik_nr.CartToJnt(seed, target, out) >= 0
        if not ok:
            ok = ik_lma.CartToJnt(seed, target, out) >= 0
        if not ok:
            raise RuntimeError(f"IK failed for {name}")
        fk_out = kdl.Frame()
        fk.JntToCart(out, fk_out)
        if (target.p - fk_out.p).Norm() > IK_TOL:
            raise RuntimeError(f"IK pose error for {name}")
        return as_list(out)

    # Gen3's wrist joint is continuous: the tilt may run past pi, so it is not clamped.
    def tilted(q_pour: list[float]) -> list[float]:
        q_tilt = q_pour[:]
        q_tilt[-1] += POUR_TILT_RAD
        return q_tilt

    q_pre_pour = solve("pre-pour", HOME, kdl.Vector(JUG_X, JUG_Y, TABLE_Z + 0.27))
    pour_pos = kdl.Vector(JUG_X, JUG_Y, TABLE_Z + 0.20)
    q_pour = solve("pour", q_pre_pour, pour_pos)
    q_tilt = tilted(q_pour)
    # Tilting swings the outlet away from where the untilted pose put it; shift the pour pose
    # until the tilted outlet sits over the receiver.
    for _ in range(4):
        robot.set_joint_pos(q_tilt, call_forward=False)
        outlet = env.site_frame("pour_outlet").p
        err = kdl.Vector(JUG_X, JUG_Y, TILT_OUTLET_Z) - outlet
        if err.Norm() < 5e-3:
            break
        pour_pos = pour_pos + err
        q_pour = solve("pour", q_pre_pour, pour_pos)
        q_tilt = tilted(q_pour)
    q_retreat = solve("retreat", q_pour, kdl.Vector(RETREAT_X, RETREAT_Y, TABLE_Z + 0.27))
    return {
        "home": HOME[:],
        "pre_pour": q_pre_pour,
        "pour": q_pour,
        "tilt": q_tilt,
        "retreat": q_retreat,
    }


def apply_pd_gravity(robot: mjk.Robot, target: list[float]) -> None:
    robot.update()
    gravity = robot.gravity_torques(-9.81)
    robot.jnt_trq_cmd = [
        KP[i] * (target[i] - robot.jnt_pos_msr[i]) - KD[i] * robot.jnt_vel_msr[i] + gravity[i]
        for i in range(robot.n_joints)
    ]
    robot.update()


def max_abs_joint_err(robot: mjk.Robot, target: list[float]) -> float:
    return max(abs(target[i] - robot.jnt_pos_msr[i]) for i in range(robot.n_joints))


def lerp(start: list[float], target: list[float], alpha: float) -> list[float]:
    return [a + alpha * (b - a) for a, b in zip(start, target)]


def place_balls_in_bottle(env: mjk.Env, robot: mjk.Robot) -> None:
    robot.set_joint_pos(HOME, call_forward=True)
    center = env.site_frame("pour_center")
    spacing = 2.0 * BALL_RADIUS
    for i in range(NUM_BALLS):
        layer = i // 9
        slot = i % 9
        ix = float(slot % 3) - 1.0
        iy = float(slot // 3) - 1.0
        local = kdl.Vector(ix * spacing, iy * spacing, -0.026 + layer * spacing)
        world = center * local
        env.set_body_pose(f"grain_{i:02d}", [world.x(), world.y(), world.z()])


def balls_in_receiver(env: mjk.Env) -> tuple[int, list[float]]:
    count = 0
    centroid = [0.0, 0.0, 0.0]
    for i in range(NUM_BALLS):
        frame = env.body_frame(f"grain_{i:02d}")
        pos = [frame.p.x(), frame.p.y(), frame.p.z()]
        centroid = [a + b for a, b in zip(centroid, pos)]
        if (
            abs(pos[0] - JUG_X) < JUG_RADIUS - 0.012
            and abs(pos[1] - JUG_Y) < JUG_RADIUS - 0.012
            and TABLE_Z + 0.006 < pos[2] < TABLE_Z + JUG_HEIGHT + 0.04
        ):
            count += 1
    return count, [value / NUM_BALLS for value in centroid]


def step_once(robot: mjk.Robot, viewer: mjk.SimulateViewer | None) -> bool:
    if viewer is None:
        return robot.step()
    # step() never sleeps; without pacing the physics loop starves the render thread.
    if not viewer.step():
        return False
    viewer.pace()
    return True


def run_phase(
    env: mjk.Env,
    robot: mjk.Robot,
    phase: Phase,
    viewer: mjk.SimulateViewer | None,
    recorder: mjk.VideoRecorder | None,
    record_every: int,
    step_counter: list[int],
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
        if not step_once(robot, viewer):
            return False
        if viewer is not None and env.time() < state["prev"] - 1e-6:  # UI reset pressed
            env.reset()
            state["prev"] = env.time()
            raise ResetRequested()
        state["prev"] = env.time()
        step_counter[0] += 1
        if recorder is not None and step_counter[0] % record_every == 0:
            recorder.record_frame()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--record", nargs="?", const="table_pour.mp4")
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    env, robot = build_env()
    recorder = None
    try:
        robot.ctrl_mode = mjk.CtrlMode.TORQUE

        def on_reset(ctx):
            place_balls_in_bottle(env, robot)  # also re-homes the arm
            if env.has_actuator("g_fingers_actuator"):
                env.set_actuator_ctrl("g_fingers_actuator", GRIPPER_CLOSED)

        env.on_reset = on_reset
        env.reset()

        waypoints = build_waypoints(env, robot)
        env.reset()  # waypoint search moved the arm; start the run from home again
        g = GRIPPER_CLOSED
        phases = [
            Phase("HOME", waypoints["home"], 1.0, 2.5, 0.08, g),
            Phase("PRE_POUR", waypoints["pre_pour"], 4.0, 6.5, 0.08, g),
            Phase("POUR", waypoints["pour"], 3.5, 5.5, 0.07, g),
            Phase("TILT", waypoints["tilt"], 7.0, 10.0, 0.07, g),
            Phase("POUR_HOLD", waypoints["tilt"], 10.0 if args.gui else 9.0, 0.0, -1.0, g),
            Phase("RETREAT", waypoints["retreat"], 2.0, 4.0, 0.08, g),
            Phase("HOLD", waypoints["retreat"], 10.0 if args.gui else 1.0, 0.0, -1.0, g),
        ]

        fps = 60
        record_every = max(1, int(1.0 / (fps * env.timestep())))
        if args.record:
            recorder = mjk.VideoRecorder.open_preset(
                env, args.record, mjk.VideoResolution.R1080p, fps
            )
        step_counter = [0]
        state = {"prev": env.time()}
        if args.gui:
            viewer = mjk.SimulateViewer.open(robot, "ex_table_pour.py")
            try:
                while viewer.is_running():
                    try:
                        for phase in phases:
                            if not run_phase(
                                env, robot, phase, viewer, recorder, record_every, step_counter, state
                            ):
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
                if not run_phase(
                    env, robot, phase, None, recorder, record_every, step_counter, state
                ):
                    break
        in_receiver, centroid = balls_in_receiver(env)
        print(f"balls in transparent receiver: {in_receiver}/{NUM_BALLS}")
        print(f"grain centroid: {[round(v, 3) for v in centroid]} receiver center={[JUG_X, JUG_Y]}")
        if recorder is not None:
            print(f"recorded: {args.record}")
        if not args.gui and in_receiver < MIN_BALLS_IN_RECEIVER:
            print("pour failed: too few balls reached the receiver")
            return 1
    finally:
        if recorder is not None:
            recorder.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
