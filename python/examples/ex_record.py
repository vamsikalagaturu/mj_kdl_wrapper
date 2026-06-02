#!/usr/bin/env python3
"""Headless recording example ported from src/examples/ex_record.cpp."""

from __future__ import annotations

import os
from pathlib import Path

import mj_kdl_wrapper as mjk

DEFAULT_MODEL = "third_party/menagerie/kinova_gen3/gen3.xml"
MODEL_ENV_VAR = "MJ_KDL_MODEL"
HOME_POSE = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]


def path_from_arg(value: str, label: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{value} does not exist for {label}")
    return path


def build_scene(model_path: Path) -> tuple[mjk.Scene, mjk.Robot]:
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    robot_spec = mjk.RobotSpec()
    robot_spec.path = str(model_path)
    spec.robots = [robot_spec]
    scene = mjk.Scene.build(spec)
    robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link")
    return scene, robot


def main() -> int:
    out_path = "recording.mp4"
    fps = 30
    steps = 180
    model_path = path_from_arg(os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL), "arm model")
    scene, robot = build_scene(model_path)
    try:
        robot.ctrl_mode = mjk.CtrlMode.POSITION
        robot.set_joint_pos(HOME_POSE, call_forward=False)
        robot.jnt_pos_cmd = HOME_POSE[:]
        recorder = mjk.VideoRecorder.open_preset(scene, out_path, mjk.VideoResolution.R720p, fps)
        try:
            for _ in range(steps):
                robot.update()
                robot.step()
                recorder.record_frame()
        finally:
            recorder.close()
        print(f"recorded: {out_path}")
    finally:
        scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
