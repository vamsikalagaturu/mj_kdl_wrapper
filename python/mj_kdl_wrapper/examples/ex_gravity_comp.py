#!/usr/bin/env python3
"""KDL gravity compensation on the Gen3; Python counterpart of src/examples/ex_gravity_comp.cpp."""

from __future__ import annotations

import argparse

import PyKDL as kdl

import mj_kdl_wrapper as mjk

HOME_POSE = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
STEPS = 7500  # 15 s
MAX_DRIFT = 0.0001  # [m]


def joints(values) -> kdl.JntArray:
    q = kdl.JntArray(len(values))
    for i, value in enumerate(values):
        q[i] = value
    return q


def gravity(dyn: kdl.ChainDynParam, q_values) -> list[float]:
    g = kdl.JntArray(len(q_values))
    dyn.JntToGravity(joints(q_values), g)
    return [g[i] for i in range(g.rows())]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    robot_spec = mjk.RobotSpec()
    robot_spec.path = mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")
    spec.robots = [robot_spec]

    env = mjk.Env.build(spec)
    try:
        robot = env.create_robot("base_link", "bracelet_link")
        robot.set_control_mode(mjk.CtrlMode.TORQUE)
        chain = robot.kdl_chain()
        fk = kdl.ChainFkSolverPos_recursive(chain)
        dyn = kdl.ChainDynParam(chain, kdl.Vector(0.0, 0.0, spec.gravity_z))

        def on_reset(ctx):
            robot.set_joint_pos(HOME_POSE)
            robot.jnt_trq_cmd = gravity(dyn, HOME_POSE)

        env.on_reset = on_reset
        env.reset()
        if args.gui:
            env.open_viewer("ex_gravity_comp.py")

        start = kdl.Frame()
        fk.JntToCart(joints(HOME_POSE), start)
        for _ in range(STEPS):
            env.update()
            robot.jnt_trq_cmd = gravity(dyn, robot.jnt_pos_msr)
            if not env.step():
                break
            env.pace()
        env.update()

        end = kdl.Frame()
        fk.JntToCart(joints(robot.jnt_pos_msr), end)
        drift = (end.p - start.p).Norm()
        print(f"EE drift after {STEPS} steps: {drift * 1000:.4f} mm (limit {MAX_DRIFT * 1000} mm)")
    finally:
        env.close()
    if args.gui:
        return 0
    ok = drift <= MAX_DRIFT
    print(f"{'PASS' if ok else 'FAIL'}: the arm holds its pose")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
