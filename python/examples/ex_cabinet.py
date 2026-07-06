#!/usr/bin/env python3
"""Load the 3-drawer cabinet asset into a robot-less scene.

The cabinet is a mesh SceneObject (no robot). Headless prints each drawer's
grasp-site pose; --gui opens it in the wrapper's simulate UI. The drawers are
passive slide joints -- drag one open in the viewer and it stays put.
"""

from __future__ import annotations

import argparse

import mj_kdl_wrapper as mjk

GRASPS = ["cabinet_grasp1", "cabinet_grasp2", "cabinet_grasp3"]


def cabinet_path() -> str:
    return mjk.menagerie.asset_path("cabinet/cabinet.xml", env_var="MJ_KDL_CABINET")


def build_scene() -> mjk.Scene:
    cabinet = mjk.SceneObject()
    cabinet.name = "cabinet"
    cabinet.mjcf_path = cabinet_path()
    cabinet.pos = [0.0, 0.0, 0.0]
    cabinet.fixed = True
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    spec.objects = [cabinet]
    return mjk.Scene.build(spec)


def run_headless(scene: mjk.Scene) -> None:
    scene.step()
    for name in GRASPS:
        p = scene.site_frame(name).p
        print(f"  {name}: world xyz = ({p.x():.3f}, {p.y():.3f}, {p.z():.3f})")


def run_gui(scene: mjk.Scene) -> None:
    viewer = mjk.SimulateViewer.open(scene, "ex_cabinet.py")
    try:
        while viewer.is_running():
            if not viewer.step():
                break
    finally:
        viewer.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="open the simulate UI")
    args = parser.parse_args()

    scene = build_scene()
    try:
        if args.gui:
            run_gui(scene)
        else:
            run_headless(scene)
    finally:
        scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
