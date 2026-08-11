#!/usr/bin/env python3
"""Admittance control with an ACHD (Vereshchagin) task-space inner loop, FT-driven.

Same outer admittance loop as ex_admittance_ft.py / ex_admittance_ft_rnea.py,
but a different inner loop. Admittance is an outer force->position loop wrapped
around an inner motion controller. The siblings use an ideal POSITION loop and a
joint-space RNEA computed-torque loop; here the inner loop is TASK SPACE:

    beta = Cartesian PD on the TCP pose error (desired Cartesian acceleration)
    qddot = ACHD(q, qdot, alpha, beta)   (constrained hybrid dynamics, KDL
                                          ChainHdSolver_Vereshchagin)
    tau = RNEA(q, qdot, qddot)           (inverse dynamics for the torque)
    apply tau in TORQUE mode

The acceleration-constrained hybrid-dynamics (ACHD) solver consumes the desired
Cartesian acceleration directly, so the admittance's Cartesian target feeds it
without an IK step (alpha = identity constrains all 6 TCP DOF). It resolves the
joint accelerations through the full arm dynamics, so tracking bandwidth is
uniform across directions (a plain PD+gravity law lags/inverts the soft axis).

Outer admittance law per Cartesian axis (no position stiffness):

    M * a = F_ext - D * v
    v += a * dt          (clamped to MAX_VEL)
    offset += v * dt     (clamped to MAX_OFFSET)

The logical FT sensor sits between the Kinova wrist and the Robotiq gripper.
After closing the gripper and letting the wrist load settle, the controller
tares it (the gripper's ~10 N static load only appears once it has closed).

The run has two sources of external force, both handled by the same law:
  - Intro: a scripted force whose direction sweeps a helix (spiral_force) drives
    the admittance, so the TCP traces a helix.
  - After the helix: the scripted force stops; the controller stays in
    admittance and responds to the FT-measured force, so in the GUI you can
    ctrl + right-drag the gripper. With K = 0 there is no equilibrium to spring
    back to: when force stops, damping bleeds v -> 0 and the pose holds.
"""

from __future__ import annotations

import argparse
import math

import PyKDL as kdl
import mj_kdl_wrapper as mjk

HOME = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
TABLE_Z = 0.70

# Task-space inner-loop gains: Cartesian PD -> desired acceleration (beta), with
# per-component acceleration limits and a joint-torque clamp.
KP_LIN, KD_LIN = 5760.0, 378.0
KP_ROT, KD_ROT = 3600.0, 441.0
BETA_LIN_MAX, BETA_ROT_MAX, TAU_MAX = 3600.0, 2520.0, 212.4

# Admittance outer loop: virtual mass, damping, stiffness (isotropic).
# K_ADM = 0 -> pure hand-guiding: holds pose on release. Set > 0 to self-center.
M_ADM, D_ADM, K_ADM = 8.0, 80.0, 0.0
FORCE_DEADBAND = 2.5  # N; rejects sensor noise and settling transients
MAX_OFFSET = 0.20     # m; reachable workspace half-extent around home
MAX_VEL = 0.25        # m/s
TOOL_BODY = "g_base"  # rigid gripper base; where the headless self-check pushes
GRIPPER_ACTUATOR = "g_fingers_actuator"
SETTLE_STEPS = 300  # ~0.6 s at dt=0.002 to close the gripper before taring
HANDOFF_TARE_TIME = 1.0  # s; let scripted-motion transients settle before FT hand-guiding
SELFCHECK_PUSH = (8.0, 12.0, 6.0)

# Intro helical force: amplitude/shape and how long it is applied.
TEACH_TIME = 16.0
TEACH_RADIUS = 0.04
TEACH_RISE = 0.10
TEACH_TURNS = 5.0


def jnt(values: list[float]) -> kdl.JntArray:
    out = kdl.JntArray(len(values))
    for i, value in enumerate(values):
        out[i] = value
    return out


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def vadd(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def vscale(a: list[float], s: float) -> list[float]:
    return [s * a[i] for i in range(3)]


def vclamp(a: list[float], limit: float) -> list[float]:
    return [clamp(x, -limit, limit) for x in a]


def vnorm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def xyz(v: kdl.Vector) -> list[float]:
    return [v.x(), v.y(), v.z()]


def frame_point(frame: kdl.Frame, point: kdl.Vector) -> list[float]:
    return xyz(frame * point)


def alpha_identity() -> kdl.Jacobian:
    """Constraint matrix: all 6 TCP DOF are controlled (one beta per row)."""
    alpha = kdl.Jacobian(6)
    for i in range(6):
        alpha[i, i] = 1.0
    return alpha


def ft_attachment() -> mjk.AttachmentSpec:
    spec = mjk.AttachmentSpec()
    spec.mjcf_path = mjk.menagerie.asset_path("ft_sensor.xml", env_var="MJ_KDL_FT_SENSOR")
    spec.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
    return spec


def gripper_attachment() -> mjk.AttachmentSpec:
    spec = mjk.AttachmentSpec()
    spec.mjcf_path = mjk.menagerie.asset_path("robotiq_2f85/2f85.xml", env_var="MJ_KDL_GRIPPER")
    spec.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "wrist_ft_site")
    spec.prefix = "g_"
    return spec


def table_object() -> mjk.SceneObject:
    table = mjk.SceneObject()
    table.name = "table"
    table.mjcf_path = mjk.menagerie.asset_path("table.xml", env_var="MJ_KDL_TABLE")
    table.pos = [0.0, 0.0, TABLE_Z]
    table.fixed = True
    return table


def build_env() -> tuple[mjk.Env, mjk.Robot]:
    table = table_object()
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    spec.objects = [table]

    robot_spec = mjk.RobotSpec()
    robot_spec.path = mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")
    robot_spec.attach_to = mjk.AttachTarget(
        mjk.AttachKind.Site, mjk.scene_object_site_name(table, "table_top")
    )
    robot_spec.attachments = [ft_attachment(), gripper_attachment()]
    spec.robots = [robot_spec]

    env = mjk.Env.build(spec)

    ft = mjk.ForceTorqueSensorSpec()
    ft.name = "wrist_ft"
    ft.frame_site = "wrist_ft_site"

    tool = mjk.ToolFrameSpec()
    tool.tool_body = "g_base"
    tool.tcp_site = "g_pinch"
    tool.ft_sensors = [ft]

    robot = env.create_robot("base_link", "bracelet_link", tool=tool)
    return env, robot


def achd_track(robot: mjk.Robot, state: dict, target: kdl.Frame) -> None:
    """Inner loop: task-space control via ACHD constrained hybrid dynamics.

    A Cartesian PD on the TCP pose error is the desired acceleration (beta) for
    all 6 constrained DOF (alpha = identity); ACHD resolves it into joint
    accelerations through the arm dynamics, then RNEA maps those to torques.
    No IK is needed -- the Cartesian target feeds the solver directly.
    robot.update() must have refreshed jnt_pos_msr/jnt_vel_msr this step.
    """
    n = robot.n_joints
    q = jnt(robot.jnt_pos_msr)
    qd = jnt(robot.jnt_vel_msr)

    err = kdl.diff(robot.fk_frame(), target)
    e = [err.vel.x(), err.vel.y(), err.vel.z(), err.rot.x(), err.rot.y(), err.rot.z()]
    if state["first_pid"]:
        state["err_prev"] = e[:]
        state["first_pid"] = False
    de = [(e[i] - state["err_prev"][i]) / state["dt"] for i in range(6)]
    state["err_prev"] = e[:]

    beta = kdl.JntArray(6)
    for i in range(3):
        beta[i] = clamp(KP_LIN * e[i] + KD_LIN * de[i], -BETA_LIN_MAX, BETA_LIN_MAX)
    for i in range(3, 6):
        beta[i] = clamp(KP_ROT * e[i] + KD_ROT * de[i], -BETA_ROT_MAX, BETA_ROT_MAX)

    qdd = kdl.JntArray(n)
    ff = kdl.JntArray(n)
    constraint_tau = kdl.JntArray(n)
    f_ext = [kdl.Wrench.Zero() for _ in range(state["n_seg"])]
    if state["achd"].CartToJnt(q, qd, qdd, state["alpha"], beta, f_ext, ff, constraint_tau) < 0:
        raise RuntimeError("ACHD hybrid dynamics failed")

    tau = kdl.JntArray(n)
    if state["rnea"].CartToJnt(q, qd, qdd, f_ext, tau) < 0:
        raise RuntimeError("RNEA inverse dynamics failed")
    robot.jnt_trq_cmd = [clamp(tau[i], -TAU_MAX, TAU_MAX) for i in range(n)]


def close_gripper(env: mjk.Env) -> None:
    if env.has_actuator(GRIPPER_ACTUATOR):
        env.set_actuator_ctrl(GRIPPER_ACTUATOR, 255.0)


def settle_and_tare(env: mjk.Env, robot: mjk.Robot, state: dict) -> list[float]:
    """Close the gripper, hold the home pose until the wrist load settles, tare.

    The gripper's static load shows up at the FT site only once it has closed
    and settled (~10 N here). Taring before that (right after reset, gripper
    open) leaves a large constant bias error that an integrating (K=0)
    admittance turns into permanent drift. So we hold the closed-gripper home
    pose for a moment first, then capture the bias.
    """
    robot.update()
    home = robot.fk_frame()
    for _ in range(SETTLE_STEPS):
        robot.update()
        close_gripper(env)
        achd_track(robot, state, home)
        if not robot.step():
            break
        robot.pace()
    robot.update()
    return xyz(robot.ft_sensor_frame("wrist_ft").M * robot.ft_sensor("wrist_ft").force)


def measured_force(robot: mjk.Robot, state: dict) -> list[float]:
    """External force on the tool in world frame, gravity-tared, deadbanded.

    The MuJoCo force sensor reports the reaction wrench at the site, so the
    external push the user applies is the negated, bias-removed reading. The
    bias is the gripper's static gravity load captured after the gripper closes
    and the wrist load settles (see settle_and_tare); expressed in the world
    frame this is just the distal weight (mg, downward) and is invariant to the
    arm configuration, so a single tare stays valid as the TCP translates around
    home. Sub-deadband residue (noise, settling transients) is rejected to zero.
    """
    wrench = robot.ft_sensor("wrist_ft")
    f_world = xyz(robot.ft_sensor_frame("wrist_ft").M * wrench.force)
    bias = state["bias"]
    f_ext = [bias[i] - f_world[i] for i in range(3)]
    if vnorm(f_ext) < FORCE_DEADBAND:
        return [0.0, 0.0, 0.0]
    return f_ext


def tare_force(robot: mjk.Robot) -> list[float]:
    return xyz(robot.ft_sensor_frame("wrist_ft").M * robot.ft_sensor("wrist_ft").force)


def admittance_update(state: dict, force: list[float], dt: float) -> None:
    # offset = integral of velocity, so with K = 0 it is a pure integrator: the
    # moment the push stops (force deadbanded to zero) we kill the velocity so
    # motion stops dead and the offset (pose) is held exactly where it was left.
    if force == [0.0, 0.0, 0.0]:
        state["vel"] = [0.0, 0.0, 0.0]
        return
    acc = [
        (force[i] - D_ADM * state["vel"][i] - K_ADM * state["offset"][i]) / M_ADM
        for i in range(3)
    ]
    state["vel"] = vclamp(vadd(state["vel"], vscale(acc, dt)), MAX_VEL)
    state["offset"] = vclamp(vadd(state["offset"], vscale(state["vel"], dt)), MAX_OFFSET)


def spiral_force(t: float) -> list[float]:
    """Scripted external force whose direction sweeps a helix over TEACH_TIME.

    The force is D_ADM times the velocity of a helical path, so a mass-damper
    admittance (steady state v = F / D) turns it into helical motion. Fed into
    the admittance, this drives the intro helix.
    """
    if t < 0.0 or t > TEACH_TIME:
        return [0.0, 0.0, 0.0]
    theta = 2.0 * math.pi * TEACH_TURNS * t / TEACH_TIME
    theta_dot = 2.0 * math.pi * TEACH_TURNS / TEACH_TIME
    vx = -TEACH_RADIUS * theta_dot * math.sin(theta)
    vy = TEACH_RADIUS * theta_dot * math.cos(theta)
    vz = TEACH_RISE / TEACH_TIME
    return [D_ADM * vx, D_ADM * vy, D_ADM * vz]


def admittance_step(env, robot, nominal, state, force):
    """One admittance tick: force -> offset (outer loop) -> ACHD-tracked TCP.

    robot.update() must have run this step so the FT read behind `force` is
    current. Returns the commanded target frame (for tracing).
    """
    admittance_update(state, force, env.timestep())
    target = kdl.Frame(nominal.M, nominal.p + kdl.Vector(*state["offset"]))
    achd_track(robot, state, target)
    return target


def run_gui(env: mjk.Env, robot: mjk.Robot, nominal: kdl.Frame, state: dict) -> None:
    """Admittance control for the whole run (ACHD task-space inner loop).

    For the first TEACH_TIME seconds a scripted helical force drives the
    admittance, so the TCP traces a helix. After that the scripted force stops
    and you can ctrl + right-drag the gripper to apply your own force, which the
    FT senses; the same admittance responds and holds on release.
    """
    viewer = mjk.SimulateViewer.open(robot, "ex_admittance_ft_achd.py")
    viewer.set_free_camera(1.55, 145.0, -24.0, (0.05, 0.0, TABLE_Z + 0.35))
    prev = env.time()
    start = env.time()
    handoff_tared = False
    target_prev: list[float] | None = None
    tcp_prev: list[float] | None = None
    trace_step = 0
    try:
        while viewer.is_running():
            if env.time() < prev - 1e-6:
                env.reset()
                start = env.time()
                handoff_tared = False
                state["offset"] = [0.0, 0.0, 0.0]
                state["vel"] = [0.0, 0.0, 0.0]
                state["err_prev"] = [0.0] * 6
                state["first_pid"] = True
                target_prev = tcp_prev = None
            prev = env.time()
            t = env.time() - start
            robot.update()
            close_gripper(env)
            # Intro: the scripted helical force IS the external force the demo
            # applies (fed straight in -- also shoving the body would double-
            # actuate it). After: the FT-measured force, so a hand-drag is sensed.
            if t < TEACH_TIME:
                force = spiral_force(t)
            elif t < TEACH_TIME + HANDOFF_TARE_TIME:
                force = [0.0, 0.0, 0.0]
            else:
                if not handoff_tared:
                    state["bias"] = tare_force(robot)
                    handoff_tared = True
                force = measured_force(robot, state)
            target = admittance_step(env, robot, nominal, state, force)

            # Draw the commanded (yellow) and actual measured TCP (green) paths.
            # Both poses are in the base_link frame, so map them through the base
            # body's world pose before tracing or the trail lands at the wrong
            # place (down by the base) and looks skewed.
            trace_step += 1
            world_base = env.body_frame("base_link")
            target_xyz = frame_point(world_base, target.p)
            tcp_xyz = frame_point(world_base, robot.fk_frame().p)
            if target_prev and trace_step % 5 == 0:
                viewer.add_trace_segment(target_prev, target_xyz, (1.0, 0.95, 0.0, 1.0))
            if tcp_prev and trace_step % 5 == 0:
                viewer.add_trace_segment(tcp_prev, tcp_xyz, (0.0, 1.0, 0.2, 1.0))
            target_prev = target_xyz
            tcp_prev = tcp_xyz

            if not viewer.step():
                break
            viewer.pace()
    finally:
        env.set_body_wrench(TOOL_BODY, (0.0, 0.0, 0.0))
        viewer.close()


def run_selfcheck(env: mjk.Env, robot: mjk.Robot, nominal: kdl.Frame, state: dict) -> dict:
    """Headless exercise of the same admittance law the GUI uses. Returns metrics.

    Phase A: the scripted helical force drives the admittance (intro behaviour).
    Phase B: a physical +Y wrench is sensed by the FT and yielded to, then
    released. Verifies the admittance reacts to both force sources and holds
    when force stops.
    """
    t0 = env.time()
    helix_react = 0.0
    helix_track_err = 0.0
    while env.time() - t0 < TEACH_TIME:
        t = env.time() - t0
        robot.update()
        close_gripper(env)
        target = admittance_step(env, robot, nominal, state, spiral_force(t))
        tcp = robot.fk_frame()
        err = [tcp.p[i] - target.p[i] for i in range(3)]
        helix_react = max(helix_react, vnorm(state["offset"]))
        helix_track_err = max(helix_track_err, vnorm(err))
        if not robot.step():
            break
        robot.pace()

    handoff_force = 0.0
    t_handoff = env.time()
    while env.time() - t_handoff < HANDOFF_TARE_TIME:
        robot.update()
        close_gripper(env)
        target = admittance_step(env, robot, nominal, state, [0.0, 0.0, 0.0])
        tcp = robot.fk_frame()
        err = [tcp.p[i] - target.p[i] for i in range(3)]
        helix_track_err = max(helix_track_err, vnorm(err))
        if not robot.step():
            break
        robot.pace()
    robot.update()
    state["bias"] = tare_force(robot)
    for _ in range(100):
        robot.update()
        close_gripper(env)
        force = measured_force(robot, state)
        handoff_force = max(handoff_force, vnorm(force))
        admittance_step(env, robot, nominal, state, force)
        if not robot.step():
            break
        robot.pace()

    helix_settle_err = 0.0
    t_settle = env.time()
    while env.time() - t_settle < 0.5:
        robot.update()
        close_gripper(env)
        target = admittance_step(env, robot, nominal, state, [0.0, 0.0, 0.0])
        tcp = robot.fk_frame()
        err = [tcp.p[i] - target.p[i] for i in range(3)]
        helix_settle_err = max(helix_settle_err, vnorm(err))
        if not robot.step():
            break
        robot.pace()

    pre_push = state["offset"][:]
    t1 = env.time()
    settled: list[float] | None = None
    push_recovery_err: float | None = None
    while env.time() - t1 < 4.0:
        t = env.time() - t1
        env.set_body_wrench(TOOL_BODY, SELFCHECK_PUSH if t < 1.0 else (0.0, 0.0, 0.0))
        robot.update()
        close_gripper(env)
        target = admittance_step(env, robot, nominal, state, measured_force(robot, state))
        tcp = robot.fk_frame()
        err = [tcp.p[i] - target.p[i] for i in range(3)]
        if push_recovery_err is None and t >= 2.0:
            push_recovery_err = vnorm(err)
        # Sample once the torque loop's settle transient has died (it can ring
        # for ~1.5 s after release); hold drift is then the steady drift.
        if settled is None and t >= 2.5:
            settled = state["offset"][:]
        if not robot.step():
            break
        robot.pace()
    env.set_body_wrench(TOOL_BODY, (0.0, 0.0, 0.0))
    return {
        "helix_react": helix_react,
        "helix_track_err": helix_track_err,
        "helix_settle_err": helix_settle_err,
        "handoff_force": handoff_force,
        "push_response": vnorm([(settled or pre_push)[i] - pre_push[i] for i in range(3)]),
        "push_dy": (settled or pre_push)[1] - pre_push[1],
        "push_recovery_err": push_recovery_err or 0.0,
        "hold_drift": vnorm([state["offset"][i] - (settled or pre_push)[i] for i in range(3)]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    env, robot = build_env()
    try:
        chain = robot.kdl_chain()
        robot.ctrl_mode = mjk.CtrlMode.TORQUE  # ACHD + RNEA computed-torque inner loop

        state = {
            "bias": [0.0, 0.0, 0.0],
            "offset": [0.0, 0.0, 0.0],
            "vel": [0.0, 0.0, 0.0],
            # Vereshchagin root acceleration carries gravity as +z (its sign
            # convention); RNEA below uses the usual -z gravity.
            "achd": kdl.ChainHdSolver_Vereshchagin_Fixed_Joint(
                chain, kdl.Twist(kdl.Vector(0.0, 0.0, 9.81), kdl.Vector.Zero()), 6
            ),
            "rnea": kdl.ChainIdSolver_RNE(chain, kdl.Vector(0.0, 0.0, -9.81)),
            "alpha": alpha_identity(),
            "n_seg": chain.getNrOfSegments(),
            "err_prev": [0.0] * 6,
            "first_pid": True,
            "dt": env.timestep(),
        }

        def on_reset(ctx):
            robot.set_joint_pos(HOME, call_forward=False)
            state["offset"] = [0.0, 0.0, 0.0]
            state["vel"] = [0.0, 0.0, 0.0]
            state["err_prev"] = [0.0] * 6
            state["first_pid"] = True
            env.set_body_wrench(TOOL_BODY, (0.0, 0.0, 0.0))

        env.on_reset = on_reset
        env.reset()
        # Single tare after settling. Orientation is held during hand-guiding so
        # the world-frame gravity bias stays ~constant; a slow auto-tare would be
        # needed only if drift exceeded the deadband during large reorientations.
        state["bias"] = settle_and_tare(env, robot, state)
        nominal = robot.fk_frame()

        print(f"FT bias: [{state['bias'][0]:.3f}, {state['bias'][1]:.3f}, {state['bias'][2]:.3f}] N")
        if args.gui:
            run_gui(env, robot, nominal, state)
            print(
                "final offset: "
                f"[{state['offset'][0]:.4f}, {state['offset'][1]:.4f}, {state['offset'][2]:.4f}] m"
            )
        else:
            m = run_selfcheck(env, robot, nominal, state)
            print(f"helix force response (max offset): {m['helix_react']:.4f} m")
            print(f"helix TCP tracking error:          {m['helix_track_err']:.4f} m")
            print(f"helix settle error:                {m['helix_settle_err']:.4f} m")
            print(f"FT handoff residual force:         {m['handoff_force']:.4f} N")
            print(f"FT push response (offset norm):    {m['push_response']:.4f} m")
            print(f"FT push response (offset dY):      {m['push_dy']:.4f} m")
            print(f"push release recovery error:       {m['push_recovery_err']:.4f} m")
            print(f"hold drift after push released:    {m['hold_drift']:.4f} m")
            assert m["helix_react"] > 0.05, "admittance did not respond to the helical force"
            assert m["helix_track_err"] < 0.006, "TCP did not track the commanded helix"
            assert m["helix_settle_err"] < 0.004, "TCP did not settle cleanly after the helix"
            assert m["handoff_force"] == 0.0, "FT handoff produced a false external force"
            assert m["push_response"] > 0.05, "admittance did not yield to the FT-sensed push"
            assert m["push_recovery_err"] < 0.006, "TCP did not recover quickly after the push"
            assert m["hold_drift"] < 0.01, "pose did not hold after the push stopped"
            print("OK: admittance responded to helix + FT push and held on release")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
