#!/usr/bin/env python3
"""Render gen3 arm + Robotiq 2F-85 gripper and save docs/gen3_with_gripper.png.

Usage:
  MUJOCO_GL=egl python3 tools/render_gripper_scene.py
"""

import pathlib
import numpy as np
import mujoco
from PIL import Image

ROOT     = pathlib.Path(__file__).parent.parent
XML_PATH = ROOT / "assets" / "gen3_with_2f85.xml"
OUT_PATH = ROOT / "docs" / "gen3_with_gripper.png"


SCENE_XML = """\
<mujoco model="gripper_scene">
  <asset>
    <texture type="skybox" builtin="gradient"
             rgb1="0.3 0.45 0.65" rgb2="0.65 0.8 0.95" width="200" height="200"/>
    <texture type="2d" name="groundplane" builtin="checker"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texrepeat="5 5" reflectance="0.15"/>
  </asset>
  <worldbody>
    <light pos="1.5 -1 3" dir="-0.3 0.4 -1" directional="true" diffuse="0.8 0.8 0.75"/>
    <light pos="-2 2 2.5" dir="0.5 -0.5 -0.8" directional="true" diffuse="0.4 0.45 0.55"/>
    <geom name="floor" type="plane" material="groundplane" size="5 5 0.05"/>
  </worldbody>
  <include file="{xml}"/>
</mujoco>
"""


def main():
    tmp = ROOT / "docs" / "_tmp_gripper_scene.xml"
    tmp.write_text(SCENE_XML.format(xml=str(XML_PATH)))
    print(f"Loading {XML_PATH} ...")
    model = mujoco.MjModel.from_xml_path(str(tmp))
    tmp.unlink()
    data  = mujoco.MjData(model)

    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i+1}")
                 for i in range(7)]
    for jid, rad in zip(joint_ids, [0.0, -0.4, 0.0, 1.4, 0.0, -1.0, 0.0]):
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = rad
    mujoco.mj_forward(model, data)

    W, H = 1280, 720
    model.vis.global_.offwidth  = W
    model.vis.global_.offheight = H
    model.vis.headlight.ambient[:]  = [0.4, 0.4, 0.4]
    model.vis.headlight.diffuse[:]  = [0.7, 0.7, 0.7]
    model.vis.headlight.specular[:] = [0.1, 0.1, 0.1]
    model.vis.headlight.active      = 1

    renderer = mujoco.Renderer(model, height=H, width=W)

    cam           = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.6]
    cam.distance  = 1.4
    cam.azimuth   = -125.0
    cam.elevation = -15.0

    renderer.update_scene(data, camera=cam, scene_option=mujoco.MjvOption())
    pixels = renderer.render()

    OUT_PATH.parent.mkdir(exist_ok=True)
    Image.fromarray(pixels).save(str(OUT_PATH))
    print(f"Saved {OUT_PATH}  ({W}x{H})")


if __name__ == "__main__":
    main()
