#!/usr/bin/env python3
"""Headless recording example ported from src/examples/ex_record.cpp.

Records the Kinova GEN3 arm holding the home pose via KDL gravity compensation
(TORQUE mode) for 5 s at 60 fps, with the camera slowly orbiting the arm. No
display or GLFW window is needed.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import mj_kdl_wrapper as mjk

DEFAULT_MODEL = "third_party/menagerie/kinova_gen3/gen3.xml"
MODEL_ENV_VAR = "MJ_KDL_MODEL"
HOME_POSE = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
FPS = 60
DURATION = 5.0  # seconds
GRAVITY_Z = -9.81

RESOLUTIONS = {
    "360p": mjk.VideoResolution.R360p,
    "480p": mjk.VideoResolution.R480p,
    "720p": mjk.VideoResolution.R720p,
    "1080p": mjk.VideoResolution.R1080p,
    "2k": mjk.VideoResolution.R2K,
    "4k": mjk.VideoResolution.R4K,
}


def resolve_model(path: str) -> Path:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} does not exist; set {MODEL_ENV_VAR} or run from repo root"
        )
    return model_path


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
    out_path = "sim.mp4"
    resolution = mjk.VideoResolution.R1080p
    for arg in sys.argv[1:]:
        if arg.endswith(".mp4"):
            out_path = arg
        elif arg in RESOLUTIONS:
            resolution = RESOLUTIONS[arg]
        else:
            print(f"Unknown argument '{arg}'; using 1080p", file=sys.stderr)

    model_path = resolve_model(os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL))
    scene, robot = build_scene(model_path)
    recorder = None
    try:
        # Hold the home pose with KDL gravity compensation in TORQUE mode.
        robot.ctrl_mode = mjk.CtrlMode.TORQUE
        robot.set_joint_pos(HOME_POSE, call_forward=True)
        # Prime gravity torques so the first step is compensated.
        robot.jnt_trq_cmd = list(robot.gravity_torques(GRAVITY_Z))

        recorder = mjk.VideoRecorder.open_preset(scene, out_path, resolution, FPS)
        # Orbit camera slowly around the arm for a nicer recording.
        recorder.set_free_camera(distance=1.8, azimuth=0.0, elevation=-20.0, lookat=(0.0, 0.0, 0.5))

        total_steps = int(DURATION / scene.timestep())
        steps_per_frame = max(1, int(1.0 / (FPS * scene.timestep())))
        step_per_deg = 360.0 / total_steps  # full orbit over DURATION

        print(f"Recording {DURATION} s to {out_path} ({total_steps} steps, {FPS} fps)...")
        for step in range(total_steps):
            robot.update()
            robot.jnt_trq_cmd = list(robot.gravity_torques(GRAVITY_Z))
            robot.step()
            if step % steps_per_frame == 0:
                recorder.set_free_camera(
                    distance=1.8,
                    azimuth=math.fmod(step * step_per_deg, 360.0),
                    elevation=-20.0,
                    lookat=(0.0, 0.0, 0.5),
                )
                if not recorder.record_frame():
                    print(f"record_frame() failed at step {step}", file=sys.stderr)
                    break
        print(f"Saved: {out_path}")
    finally:
        if recorder is not None:
            recorder.close()
        scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
