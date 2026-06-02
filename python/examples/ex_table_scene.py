#!/usr/bin/env python3
"""Table-scene example ported from src/examples/ex_table_scene.cpp."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import mj_kdl_wrapper as mjk

SURFACE_Z = 0.7
DEFAULT_MODEL = "third_party/menagerie/kinova_gen3/gen3.xml"
DEFAULT_GRIPPER = "third_party/menagerie/robotiq_2f85/2f85.xml"
DEFAULT_TABLE = "src/examples/assets/table.xml"
MODEL_ENV_VAR = "MJ_KDL_MODEL"
GRIPPER_ENV_VAR = "MJ_KDL_GRIPPER"
TABLE_ENV_VAR = "MJ_KDL_TABLE"
HOME_POSE = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]


def path_from_arg(value: str, label: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{value} does not exist for {label}")
    return path


def attachment_gripper(path: Path) -> mjk.AttachmentSpec:
    spec = mjk.AttachmentSpec()
    spec.mjcf_path = str(path)
    spec.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
    spec.prefix = "g_"
    return spec


def table_object(path: Path) -> mjk.SceneObject:
    obj = mjk.SceneObject()
    obj.name = "table"
    obj.mjcf_path = str(path)
    obj.pos = [0.0, 0.0, SURFACE_Z]
    obj.fixed = True
    return obj


def make_box(name, x, y, hx, hy, hz, color) -> mjk.SceneObject:
    obj = mjk.SceneObject()
    obj.name = name
    obj.shape = mjk.Shape.BOX
    obj.size = [hx, hy, hz]
    obj.pos = [x, y, SURFACE_Z + hz]
    obj.rgba = [color[0], color[1], color[2], 1.0]
    obj.mass = 0.2
    obj.friction = [1.0, 0.005, 0.0001]
    return obj


def make_sphere(name, x, y, radius, color) -> mjk.SceneObject:
    obj = mjk.SceneObject()
    obj.name = name
    obj.shape = mjk.Shape.SPHERE
    obj.size = [radius, 0.0, 0.0]
    obj.pos = [x, y, SURFACE_Z + radius]
    obj.rgba = [color[0], color[1], color[2], 1.0]
    obj.mass = 0.1
    obj.friction = [1.0, 0.005, 0.0001]
    return obj


def scene_objects(table_path):
    return [
        table_object(table_path),
        make_box("red_cube", 0.35, 0.10, 0.03, 0.03, 0.03, (1.0, 0.2, 0.2)),
        make_box("green_cube", 0.35, -0.10, 0.03, 0.03, 0.03, (0.2, 1.0, 0.2)),
        make_box("blue_cube", 0.35, 0.30, 0.04, 0.04, 0.04, (0.2, 0.2, 1.0)),
        make_sphere("orange_sphere", -0.20, 0.20, 0.035, (1.0, 0.55, 0.0)),
        make_sphere("purple_sphere", -0.20, -0.20, 0.025, (0.7, 0.0, 0.9)),
    ]


def build_scene(
    model_path: Path, gripper_path: Path, table_path: Path
) -> tuple[mjk.Scene, mjk.Robot]:
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    spec.objects = scene_objects(table_path)
    robot_spec = mjk.RobotSpec()
    robot_spec.path = str(model_path)
    robot_spec.pos = [0.0, 0.0, SURFACE_Z]
    robot_spec.attachments = [attachment_gripper(gripper_path)]
    spec.robots = [robot_spec]
    overview = mjk.CameraSpec()
    overview.name = "overview"
    overview.pos = [0.0, -0.6, 1.6]
    overview.euler = [34.0, 0.0, 0.0]
    overview.fovy = 45.0
    side = mjk.CameraSpec()
    side.name = "side"
    side.pos = [-1.0, 0.0, 1.1]
    side.euler = [0.0, -68.0, 0.0]
    side.fovy = 45.0
    spec.cameras = [overview, side]
    scene = mjk.Scene.build(spec)
    tool = mjk.ToolFrameSpec()
    tool.tool_body = "g_base"
    tool.tcp_site = "g_pinch"
    robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link", tool=tool)
    return scene, robot


def run_loop(scene: mjk.Scene, robot: mjk.Robot, step_fn, *, duration: float, gui: bool) -> None:
    if gui:
        viewer = mjk.SimulateViewer.open(robot, "ex_table_scene.py")
        try:
            while viewer.is_running():
                step_fn()
                if not viewer.step():
                    break
        finally:
            viewer.close()
        return
    end = scene.time() + duration
    while scene.time() < end:
        step_fn()
        robot.step()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    scene, robot = build_scene(
        path_from_arg(os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL), "arm model"),
        path_from_arg(os.environ.get(GRIPPER_ENV_VAR, DEFAULT_GRIPPER), "gripper model"),
        path_from_arg(os.environ.get(TABLE_ENV_VAR, DEFAULT_TABLE), "table model"),
    )
    try:
        robot.ctrl_mode = mjk.CtrlMode.TORQUE
        robot.set_joint_pos(HOME_POSE, call_forward=False)
        top = scene.site_frame(mjk.scene_object_site_name(scene.spec.objects[0], "table_top"))
        print(f"table top z = {top.p.z():.3f}")
        print(f"cameras: {' '.join(scene.camera_names())}")

        def step():
            robot.update()
            robot.jnt_trq_cmd = robot.gravity_torques(-9.81)
            if scene.has_actuator("g_fingers_actuator"):
                scene.set_actuator_ctrl(
                    "g_fingers_actuator", 255.0 if math.fmod(scene.time(), 6.0) < 3.0 else 0.0
                )

        run_loop(scene, robot, step, duration=1.0, gui=args.gui)
    finally:
        scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
