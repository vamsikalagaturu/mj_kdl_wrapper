# 3-Drawer Cabinet Asset

A stackable 3-drawer desk organizer for MuJoCo / mj_kdl_wrapper. The frame is
fixed; each drawer is a free body resting on simple support shelves, so it can
be pulled out the front and fall.
Every drawer carries a 96 mm bar handle and a grasp site at the handle center.
`cabinet.xml` is a single-root-body asset, so it drops straight into a
`mj_kdl_wrapper` scene as a `SceneObject`. Units are meters.

## Files

| File | Contents |
|------|----------|
| `cabinet.xml`            | The model (single root body `cabinet`, 3 free-body drawers) |
| `cabinet_frame_top.stl`  | Top frame module |
| `cabinet_frame_mid.stl`  | Middle frame module (instanced twice) |
| `cabinet_frame_bottom.stl` | Base plate |
| `cabinet_drawer.stl`     | Drawer (instanced three times) |
| `handle_body_96mm.stl`   | Handle main body |
| `handle_cover_96mm.stl`  | Handle cover plate |

## Names (as attached with object name `cabinet`)

Elements are prefixed `cabinet_`: drawer bodies `cabinet_drawer1..3` (free
bodies guided by translucent support shelves, side rails, and back stops) and
grasp sites `cabinet_grasp1..3` at the handle centers (invisible; for pose
queries). Drag a drawer forward in the viewer and it can leave the cabinet and
fall.

## Use it in mj_kdl_wrapper

See `python/examples/ex_cabinet.py`. In short:

```python
import mj_kdl_wrapper as mjk

obj = mjk.SceneObject()
obj.name, obj.mjcf_path, obj.pos, obj.fixed = "cabinet", ".../cabinet.xml", [0, 0, 0], True

spec = mjk.SceneSpec()
spec.timestep, spec.add_floor, spec.add_skybox, spec.objects = 0.002, True, True, [obj]
scene = mjk.Scene.build(spec)

scene.step()
pose = scene.site_frame("cabinet_grasp1")     # PyKDL.Frame at the handle center

# pull drawer 1 out the +X front:
scene.set_body_wrench("cabinet_drawer1", [40.0, 0.0, 0.0])
for _ in range(300):
    scene.step()
```

## Standalone view

```bash
python -m mujoco.viewer --mjcf=cabinet.xml
```

## Notes

- **Mesh visuals do not collide.** MuJoCo collides meshes as convex hulls, so
  the asset uses simple translucent box geoms for drawer, shelf, side-rail, and
  back-stop contact.
- Handles have explicit translucent capsule collision geoms for the grasp bar
  and mounting posts. Use the grasp sites for approach targets; the capsule
  geoms provide contact for closing fingers around the handle.
- The cabinet collision is deliberately simple. Add more detailed collision
  geometry if a robot needs accurate contact with the frame.

## Attribution

STL meshes are the original authors' work, from MakerWorld:

- **Cabinet / drawers** — *Office Desk Organizer - 3 Drawer Storage*
  https://makerworld.com/en/models/2500817-office-desk-organizer-3-drawer-storage
- **Handles** — *Dual-Color Furniture Handle / Drawer Handle*
  https://makerworld.com/en/models/2431804-dual-color-furniture-handle-drawer-handle

Check each model's MakerWorld license before redistributing the meshes.
