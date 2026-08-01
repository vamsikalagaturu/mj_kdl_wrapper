# Documentation {#page_documentation}

- [C++ Tutorial](@ref page_tutorial_cpp)
- [Python Tutorial](@ref page_tutorial_python)
- [C++ API Guide](@ref page_api_cpp)
- [Python Bindings API Guide](@ref page_api_python)
- [Examples](@ref page_examples)
- [Torque Control and Tool Inertia](@ref page_howto_torque_control)
- [Importing a URDF Robot](@ref page_howto_urdf)

## Migrating from 0.2.x {#sec_migrate_quat}

0.3.0 removes the `euler` placement field from `RobotSpec`, `AttachmentSpec`,
`SceneObject` and `CameraSpec`, and from `attach_child()`. Placement orientation
is now the quaternion `quat`, in `[x, y, z, w]` order, with identity
`{ 0, 0, 0, 1 }`. There is no compatibility shim: code setting `euler` no longer
compiles.

`quat` is the library's own convention. MuJoCo's `[w, x, y, z]` ordering stays
inside the wrapper and is never exposed through these specs.

C++, a 180-degree flip about x:

```cpp
// 0.2.x
cam.euler = { 180.0, 0.0, 0.0 };
// 0.3.0
cam.quat = { 1.0, 0.0, 0.0, 0.0 };
```

Python, the same rotation:

```python
# 0.2.x
cam.euler = [180.0, 0.0, 0.0]
# 0.3.0
cam.quat = [1.0, 0.0, 0.0, 0.0]
```

For angles without an exact quaternion, convert with SciPy --
`Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_quat()` returns
`[x, y, z, w]` directly.
