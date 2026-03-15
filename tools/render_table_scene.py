#!/usr/bin/env python3
"""Render the table scene offscreen and save docs/table_scene_screenshot.png.

Build and export with:
  cd build && make export_table_scene
  ./build/export_table_scene assets/gen3_urdf/GEN3_URDF_V12.urdf /tmp/table_scene.mjb
Run with:
  MUJOCO_GL=egl python3 tools/render_table_scene.py
"""

import pathlib
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT     = pathlib.Path(__file__).parent.parent
MJB_PATH = pathlib.Path("/tmp/table_scene.mjb")
OUT_PATH = ROOT / "docs" / "table_scene_screenshot.png"


def export_model():
    binary = ROOT / "build" / "export_table_scene"
    urdf   = ROOT / "assets" / "gen3_urdf" / "GEN3_URDF_V12.urdf"
    if not binary.exists():
        sys.exit(f"Export binary not found: {binary}\n"
                 f"Run: cd build && make export_table_scene")
    r = subprocess.run([str(binary), str(urdf), str(MJB_PATH)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"export_table_scene failed:\n{r.stderr}")
    print(r.stdout.strip())


def main():
    if not MJB_PATH.exists():
        print(f"{MJB_PATH} not found — running export_table_scene ...")
        export_model()

    print(f"Loading {MJB_PATH} ...")
    model = mujoco.MjModel.from_binary_path(str(MJB_PATH))
    data  = mujoco.MjData(model)

    # Home pose
    home_rad = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i+1}")
                 for i in range(7)]
    for jid, rad in zip(joint_ids, home_rad):
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = rad
    mujoco.mj_forward(model, data)

    model.vis.headlight.ambient[:]  = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:]  = [0.9,  0.9,  0.9 ]
    model.vis.headlight.specular[:] = [0.15, 0.15, 0.15]
    model.vis.headlight.active      = 1

    W, H = 1280, 720
    model.vis.global_.offwidth  = W
    model.vis.global_.offheight = H
    renderer = mujoco.Renderer(model, height=H, width=W)

    cam           = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.05, 0.0, 0.82]
    cam.distance  = 2.2
    cam.azimuth   = -140.0
    cam.elevation = -24.0

    opt = mujoco.MjvOption()
    renderer.update_scene(data, camera=cam, scene_option=opt)

    scn = renderer.scene
    max_light = len(scn.lights)
    for spec_light in [
        dict(pos=[1.5,-1.0,3.0], dir=[-0.3,0.4,-1.0],
             diffuse=[0.8,0.8,0.75], specular=[0.3,0.3,0.3]),
        dict(pos=[-2.0,2.0,2.5], dir=[0.5,-0.5,-0.8],
             diffuse=[0.35,0.4,0.5], specular=[0.0,0.0,0.0]),
    ]:
        if scn.nlight >= max_light:
            break
        l = scn.lights[scn.nlight]
        l.pos[:]      = spec_light["pos"]
        l.dir[:]      = spec_light["dir"]
        l.diffuse[:]  = spec_light["diffuse"]
        l.specular[:] = spec_light["specular"]
        l.ambient[:]  = [0.0, 0.0, 0.0]
        l.type        = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        scn.nlight   += 1

    pixels = renderer.render()

    OUT_PATH.parent.mkdir(exist_ok=True)
    Image.fromarray(pixels).save(str(OUT_PATH))
    print(f"Saved  {OUT_PATH}  ({W}x{H})")


if __name__ == "__main__":
    main()
