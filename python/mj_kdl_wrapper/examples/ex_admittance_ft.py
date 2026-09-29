#!/usr/bin/env python3
"""F/T admittance around an RNEA task-space inner loop; Python counterpart of ex_admittance_ft.cpp.

Headless, it runs the self-check and exits 1 when a metric is out of its limit; with --gui the
same sequence runs once, the mouse (ctrl + right-drag) pushing the tool after the helix.
"""

from __future__ import annotations

import argparse
import math

import PyKDL as kdl
import mj_kdl_wrapper as mjk

HOME = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
TABLE_Z = 0.70

# Inner loop: Cartesian PD gains [1/s^2], [1/s] and acceleration limits.
KP_LIN, KD_LIN = 2500.0, 100.0
KP_ROT, KD_ROT = 2500.0, 100.0
BETA_LIN_MAX, BETA_ROT_MAX = 300.0, 300.0
KP_NULL, KD_NULL = 100.0, 20.0  # posture gains in the task's null space

# Admittance: virtual mass, damping, stiffness (isotropic); K = 0 holds the pose on release.
M_ADM, D_ADM, K_ADM = 8.0, 80.0, 0.0
FORCE_DEADBAND = 2.5  # N; rejects sensor noise and settling transients
MAX_OFFSET = 0.20     # m; reachable workspace half-extent around home
MAX_VEL = 0.25        # m/s
TOOL_BODY = "g_base"  # rigid gripper base; where the headless self-check pushes
GRIPPER_ACTUATOR = "g_fingers_actuator"
FT_SITE = "wrist_ft_site"  # the F/T sensor's frame
GRIPPER_CLOSED = 0.82  # rad; driver joint, the bundled 2F-85's ctrlrange is 0..0.82
SETTLE_STEPS = 300  # ~0.6 s at dt=0.002 to close the gripper before taring
HANDOFF_TARE_TIME = 1.0  # s; let scripted-motion transients settle before FT hand-guiding
GUIDE_TIME = 4.7  # s; hand-guiding window, as long as the self-check's push phases
SELFCHECK_PUSH = (8.0, 12.0, 6.0)
ELBOW_BODY = "forearm_link"  # its origin is the elbow joint
MIN_ELBOW_HEIGHT = 0.64  # m above the table

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
    robot_spec.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "table_top")
    robot_spec.attachments = [ft_attachment(), gripper_attachment()]
    spec.robots = [robot_spec]

    env = mjk.Env.build(spec)

    ft = mjk.ForceTorqueSensorSpec()
    ft.name = "wrist_ft"
    ft.frame_site = FT_SITE

    tool = mjk.ToolFrameSpec()
    tool.tool_body = "g_base_mount"
    tool.tcp_site = "g_pinch"
    tool.ft_sensors = [ft]

    robot = env.create_robot("base_link", "bracelet_link", tool=tool)
    return env, robot


def tcp_frame(robot: mjk.Robot, state: dict) -> kdl.Frame:
    frame = kdl.Frame()
    state["fk"].JntToCart(jnt(robot.jnt_pos_msr), frame)
    return frame


def jacobian_twist(jac: kdl.Jacobian, qdot: kdl.JntArray) -> list[float]:
    return [sum(jac[row, col] * qdot[col] for col in range(qdot.rows())) for row in range(6)]


def rnea_track(env: mjk.Env, robot: mjk.Robot, state: dict, target: kdl.Frame) -> None:
    """Task-space computed torque: Cartesian PD -> qddot + null-space posture -> RNEA torque."""
    q = jnt(robot.jnt_pos_msr)
    qdot = jnt(robot.jnt_vel_msr)

    err = kdl.diff(tcp_frame(robot, state), target)
    jac = kdl.Jacobian(robot.n_joints)
    state["jac_solver"].JntToJac(q, jac)
    tcp_vel = jacobian_twist(jac, qdot)

    qddot = kdl.JntArray(robot.n_joints)
    beta = kdl.Twist()
    beta.vel = kdl.Vector(
        clamp(KP_LIN * err.vel.x() - KD_LIN * tcp_vel[0], -BETA_LIN_MAX, BETA_LIN_MAX),
        clamp(KP_LIN * err.vel.y() - KD_LIN * tcp_vel[1], -BETA_LIN_MAX, BETA_LIN_MAX),
        clamp(KP_LIN * err.vel.z() - KD_LIN * tcp_vel[2], -BETA_LIN_MAX, BETA_LIN_MAX),
    )
    beta.rot = kdl.Vector(
        clamp(KP_ROT * err.rot.x() - KD_ROT * tcp_vel[3], -BETA_ROT_MAX, BETA_ROT_MAX),
        clamp(KP_ROT * err.rot.y() - KD_ROT * tcp_vel[4], -BETA_ROT_MAX, BETA_ROT_MAX),
        clamp(KP_ROT * err.rot.z() - KD_ROT * tcp_vel[5], -BETA_ROT_MAX, BETA_ROT_MAX),
    )
    # qdd = J#(beta - J z) + z = J# beta + (I - J# J) z (Siciliano et al. 2009, Sec. 3.5.1).
    z = jnt([KP_NULL * (HOME[i] - q[i]) - KD_NULL * qdot[i] for i in range(robot.n_joints)])
    jz = jacobian_twist(jac, z)
    beta = beta - kdl.Twist(kdl.Vector(*jz[:3]), kdl.Vector(*jz[3:]))
    if state["acc_ik"].CartToJnt(q, beta, qddot) < 0:
        raise RuntimeError("RNEA task acceleration solve failed")
    for i in range(robot.n_joints):
        qddot[i] += z[i]

    tau = kdl.JntArray(robot.n_joints)
    wrenches = [kdl.Wrench.Zero() for _ in range(state["n_seg"])]
    if state["id_solver"].CartToJnt(q, qdot, qddot, wrenches, tau) < 0:
        raise RuntimeError("RNEA inverse dynamics failed")
    robot.jnt_trq_cmd = [tau[i] for i in range(robot.n_joints)]


def close_gripper(env: mjk.Env) -> None:
    env.data.actuator(GRIPPER_ACTUATOR).ctrl[0] = GRIPPER_CLOSED


def settle_and_tare(env: mjk.Env, robot: mjk.Robot, state: dict) -> list[float]:
    """Hold home while the gripper closes, then tare: its ~10 N load only appears once closed."""
    env.update()
    home = tcp_frame(robot, state)
    for _ in range(SETTLE_STEPS):
        env.update()
        close_gripper(env)
        rnea_track(env, robot, state, home)
        if not env.step():
            break
        env.pace()
    env.update()
    return tare_force(env, robot)


def measured_force(env: mjk.Env, robot: mjk.Robot, state: dict) -> list[float]:
    """External force on the tool in the world frame: tared reaction, negated, deadbanded."""
    wrench = robot.ft_sensor("wrist_ft")
    f_world = xyz(env.site_frame(FT_SITE).M * wrench.force)
    bias = state["bias"]
    f_ext = [bias[i] - f_world[i] for i in range(3)]
    force_norm = vnorm(f_ext)
    if force_norm < FORCE_DEADBAND:
        return [0.0, 0.0, 0.0]
    return f_ext


def tare_force(env: mjk.Env, robot: mjk.Robot) -> list[float]:
    return xyz(env.site_frame(FT_SITE).M * robot.ft_sensor("wrist_ft").force)


def admittance_update(state: dict, force: list[float], dt: float) -> None:
    # With K = 0 the offset integrates velocity, so no force stops the motion where it is.
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
    """D_ADM times a helix's velocity: the mass-damper admittance (v = F / D) traces the helix."""
    if t < 0.0 or t > TEACH_TIME:
        return [0.0, 0.0, 0.0]
    theta = 2.0 * math.pi * TEACH_TURNS * t / TEACH_TIME
    theta_dot = 2.0 * math.pi * TEACH_TURNS / TEACH_TIME
    vx = -TEACH_RADIUS * theta_dot * math.sin(theta)
    vy = TEACH_RADIUS * theta_dot * math.cos(theta)
    vz = TEACH_RISE / TEACH_TIME
    return [D_ADM * vx, D_ADM * vy, D_ADM * vz]


def admittance_step(env, robot, nominal, state, force):
    """Force -> offset (outer loop) -> RNEA-tracked TCP target, which it returns."""
    admittance_update(state, force, env.model.opt.timestep)
    target = kdl.Frame(nominal.M, nominal.p + kdl.Vector(*state["offset"]))
    rnea_track(env, robot, state, target)
    return target


def run_gui(env: mjk.Env, robot: mjk.Robot, nominal: kdl.Frame, state: dict) -> None:
    """The same sequence with the viewer; after the helix the mouse pushes the tool."""
    env.open_viewer("ex_admittance_ft.py")
    viewer = env.viewer
    viewer.set_free_camera(1.55, 145.0, -24.0, (0.05, 0.0, TABLE_Z + 0.35))
    start = env.data.time
    handoff_tared = False
    target_prev: list[float] | None = None
    tcp_prev: list[float] | None = None
    trace_step = 0
    try:
        while env.data.time - start < TEACH_TIME + HANDOFF_TARE_TIME + GUIDE_TIME:
            # on_reset flags a UI reset and has already re-seeded the admittance state.
            if state["reset"]:
                state["reset"] = False
                start = env.data.time
                handoff_tared = False
                target_prev = tcp_prev = None
                viewer.clear_trace()
            t = env.data.time - start
            env.update()
            close_gripper(env)
            # The helix force is fed to the admittance directly, not applied to the body.
            if t < TEACH_TIME:
                force = spiral_force(t)
            elif t < TEACH_TIME + HANDOFF_TARE_TIME:
                force = [0.0, 0.0, 0.0]
            else:
                if not handoff_tared:
                    state["bias"] = tare_force(env, robot)
                    handoff_tared = True
                force = measured_force(env, robot, state)
            target = admittance_step(env, robot, nominal, state, force)

            # Commanded (yellow) and measured (green) TCP paths; the chain's frames are base_link's.
            trace_step += 1
            world_base = env.body_frame("base_link")
            target_xyz = frame_point(world_base, target.p)
            tcp_xyz = frame_point(world_base, tcp_frame(robot, state).p)
            if target_prev and trace_step % 5 == 0:
                viewer.add_trace_segment(target_prev, target_xyz, (1.0, 0.95, 0.0, 1.0))
            if tcp_prev and trace_step % 5 == 0:
                viewer.add_trace_segment(tcp_prev, tcp_xyz, (0.0, 1.0, 0.2, 1.0))
            target_prev = target_xyz
            tcp_prev = tcp_xyz

            if not env.step():
                break
            env.pace()
    finally:
        env.data.body(TOOL_BODY).xfrc_applied[:] = 0.0


def elbow_height(env: mjk.Env) -> float:
    return env.body_frame(ELBOW_BODY).p.z() - TABLE_Z


def run_selfcheck(env: mjk.Env, robot: mjk.Robot, nominal: kdl.Frame, state: dict) -> dict:
    """The helix, the tare, then a scripted push on the tool sensed by the F/T; returns metrics."""
    elbow_start = elbow_min = elbow_height(env)
    t0 = env.data.time
    helix_react = 0.0
    helix_track_err = 0.0
    while env.data.time - t0 < TEACH_TIME:
        t = env.data.time - t0
        env.update()
        close_gripper(env)
        target = admittance_step(env, robot, nominal, state, spiral_force(t))
        tcp = tcp_frame(robot, state)
        err = [tcp.p[i] - target.p[i] for i in range(3)]
        helix_react = max(helix_react, vnorm(state["offset"]))
        helix_track_err = max(helix_track_err, vnorm(err))
        if not env.step():
            break
        elbow_min = min(elbow_min, elbow_height(env))
        env.pace()

    handoff_force = 0.0
    t_handoff = env.data.time
    while env.data.time - t_handoff < HANDOFF_TARE_TIME:
        env.update()
        close_gripper(env)
        target = admittance_step(env, robot, nominal, state, [0.0, 0.0, 0.0])
        tcp = tcp_frame(robot, state)
        err = [tcp.p[i] - target.p[i] for i in range(3)]
        helix_track_err = max(helix_track_err, vnorm(err))
        if not env.step():
            break
        elbow_min = min(elbow_min, elbow_height(env))
        env.pace()
    env.update()
    state["bias"] = tare_force(env, robot)
    for _ in range(100):
        env.update()
        close_gripper(env)
        force = measured_force(env, robot, state)
        handoff_force = max(handoff_force, vnorm(force))
        admittance_step(env, robot, nominal, state, force)
        if not env.step():
            break
        elbow_min = min(elbow_min, elbow_height(env))
        env.pace()

    helix_settle_err = 0.0
    t_settle = env.data.time
    while env.data.time - t_settle < 0.5:
        env.update()
        close_gripper(env)
        target = admittance_step(env, robot, nominal, state, [0.0, 0.0, 0.0])
        tcp = tcp_frame(robot, state)
        err = [tcp.p[i] - target.p[i] for i in range(3)]
        helix_settle_err = max(helix_settle_err, vnorm(err))
        if not env.step():
            break
        elbow_min = min(elbow_min, elbow_height(env))
        env.pace()

    pre_push = state["offset"][:]
    t1 = env.data.time
    settled: list[float] | None = None
    push_recovery_err: float | None = None
    while env.data.time - t1 < 4.0:
        t = env.data.time - t1
        push = SELFCHECK_PUSH if t < 1.0 else (0.0, 0.0, 0.0)
        env.data.body(TOOL_BODY).xfrc_applied[:] = [*push, 0.0, 0.0, 0.0]
        env.update()
        close_gripper(env)
        target = admittance_step(env, robot, nominal, state, measured_force(env, robot, state))
        tcp = tcp_frame(robot, state)
        err = [tcp.p[i] - target.p[i] for i in range(3)]
        if push_recovery_err is None and t >= 2.0:
            push_recovery_err = vnorm(err)
        # Once the release transient (~1.5 s) has died; the drift is judged from here to the end.
        if settled is None and t >= 2.5:
            settled = state["offset"][:]
        if not env.step():
            break
        elbow_min = min(elbow_min, elbow_height(env))
        env.pace()
    env.data.body(TOOL_BODY).xfrc_applied[:] = 0.0
    return {
        "helix_react": helix_react,
        "helix_track_err": helix_track_err,
        "helix_settle_err": helix_settle_err,
        "handoff_force": handoff_force,
        "push_response": vnorm([(settled or pre_push)[i] - pre_push[i] for i in range(3)]),
        "push_dy": (settled or pre_push)[1] - pre_push[1],
        "push_recovery_err": push_recovery_err or 0.0,
        "hold_drift": vnorm([state["offset"][i] - (settled or pre_push)[i] for i in range(3)]),
        "elbow_start": elbow_start,
        "elbow_min": elbow_min,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    env, robot = build_env()
    try:
        chain = robot.kdl_chain()
        acc_ik = kdl.ChainIkSolverVel_wdls(chain)
        acc_ik.setLambda(0.10)
        robot.set_control_mode(mjk.CtrlMode.TORQUE)  # RNEA computed-torque inner loop

        state = {
            "bias": [0.0, 0.0, 0.0],
            "offset": [0.0, 0.0, 0.0],
            "vel": [0.0, 0.0, 0.0],
            "fk": kdl.ChainFkSolverPos_recursive(chain),
            "jac_solver": kdl.ChainJntToJacSolver(chain),
            "acc_ik": acc_ik,
            "id_solver": kdl.ChainIdSolver_RNE(chain, kdl.Vector(0.0, 0.0, -9.81)),
            "n_seg": chain.getNrOfSegments(),
            "reset": False,
        }

        def on_reset(ctx):
            robot.set_joint_pos(HOME)
            state["offset"] = [0.0, 0.0, 0.0]
            state["vel"] = [0.0, 0.0, 0.0]
            env.data.body(TOOL_BODY).xfrc_applied[:] = 0.0
            state["reset"] = True

        env.on_reset = on_reset
        env.reset()
        state["reset"] = False
        state["bias"] = settle_and_tare(env, robot, state)
        nominal = tcp_frame(robot, state)

        bias = state['bias']
        print(f"FT bias: [{bias[0]:.3f}, {bias[1]:.3f}, {bias[2]:.3f}] N")
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
            print(
                f"elbow height start / lowest:       {m['elbow_start']:.4f} / "
                f"{m['elbow_min']:.4f} m (limit {MIN_ELBOW_HEIGHT:.4f} m)"
            )
            ok = (
                m["helix_react"] > 0.10
                and m["helix_track_err"] < 0.006
                and m["helix_settle_err"] < 0.006
                and m["handoff_force"] == 0.0
                and m["push_response"] > 0.12
                and m["push_recovery_err"] < 0.002
                and m["hold_drift"] < 0.001
                and m["elbow_min"] >= MIN_ELBOW_HEIGHT
            )
            goal = "the admittance responded to the helix and the FT push and held on release"
            print(f"{'PASS' if ok else 'FAIL'}: {goal}")
            return 0 if ok else 1
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
