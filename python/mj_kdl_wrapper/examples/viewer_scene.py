#!/usr/bin/env python3
"""View a mj_kdl_wrapper-built scene with the official MuJoCo Python viewer for VIEW_TIME
simulated seconds."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import mj_kdl_wrapper as mjk

VIEW_TIME = 10.0  # [s] simulated


def build_scene(model_path: str) -> mjk.Env:
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    robot_spec = mjk.RobotSpec()
    robot_spec.path = model_path
    spec.robots = [robot_spec]
    return mjk.Env.build(spec)


def main() -> int:
    model_path = mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")

    env = build_scene(model_path)
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".mjb", delete=False)
        tmp.close()
        mjb_path = Path(tmp.name)
        env.save_binary(str(mjb_path))
    finally:
        env.close()

    try:
        viewer_code = """
import sys
import time
import mujoco
import mujoco.viewer

expected = sys.argv[2]
actual = mujoco.mj_versionString()
if actual != expected:
    raise SystemExit(
        f"This example exports a MuJoCo {expected} .mjb file. "
        f"Installed Python mujoco is {actual}; install the same mujoco version "
        "or rebuild the wrapper against the installed Python mujoco version."
    )

model = mujoco.MjModel.from_binary_path(sys.argv[1])
data = mujoco.MjData(model)
view_time = float(sys.argv[3])
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.monotonic()
    while viewer.is_running() and data.time < view_time:
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(max(0.0, data.time - (time.monotonic() - start)))
"""
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    viewer_code,
                    str(mjb_path),
                    mjk.mujoco_version(),
                    str(VIEW_TIME),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "MuJoCo viewer failed to start. If you are running from a sandbox, SSH session, "
                "or headless shell, run this command from a graphical session with DISPLAY or "
                "WAYLAND_DISPLAY available."
            ) from exc
    finally:
        if mjb_path.exists():
            mjb_path.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
