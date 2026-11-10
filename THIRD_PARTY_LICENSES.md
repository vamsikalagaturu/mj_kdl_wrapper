# Third-Party Licenses

This project depends on and/or vendors the following third-party components.

| Component | Use | License | Source URL |
| --- | --- | --- | --- |
| MuJoCo | Runtime dependency via the pinned `mujoco` Python package; C++ builds compile it from source and the C++ install carries `libmujoco`, its plugins and headers. | Apache License 2.0 | https://github.com/google-deepmind/mujoco |
| MuJoCo Filament renderer and Studio components | Statically linked into `libmj_kdl_render` and `libmj_kdl_viewer`; their materials and fonts are installed in `<libdir>/mujoco_assets`. | Apache License 2.0 | https://github.com/google-deepmind/mujoco/tree/main/studio |
| Filament | Statically linked into `libmj_kdl_render` and `libmj_kdl_viewer`, with the third-party libraries Filament builds with. | Apache License 2.0 | https://github.com/google/filament |
| Dear ImGui | Statically linked into `libmj_kdl_render` and `libmj_kdl_viewer`. | MIT License | https://github.com/ocornut/imgui |
| ImPlot | Statically linked into `libmj_kdl_viewer`. | MIT License | https://github.com/epezent/implot |
| SDL2 | Statically linked into `libmj_kdl_viewer`. | zlib License | https://github.com/libsdl-org/SDL |
| Atkinson Hyperlegible Next and Mono fonts | Installed in `<libdir>/mujoco_assets` with the viewer. | SIL Open Font License 1.1 | https://github.com/googlefonts/atkinson-hyperlegible-next |
| Font Awesome 4 font | Installed in `<libdir>/mujoco_assets` with the viewer. | SIL Open Font License 1.1 | https://github.com/FortAwesome/Font-Awesome |
| Orocos KDL fork | KDL library and PyKDL bindings built from the secorolab fork when this project owns the KDL build. | GNU Lesser General Public License 2.1 | https://github.com/secorolab/orocos_kinematics_dynamics |
| Robotiq 2F-85 Menagerie asset | `assets/robotiq_2f85/` is derived from MuJoCo Menagerie's Robotiq 2F-85 model and modified for this project. | BSD 2-Clause style license, copyright ROS-Industrial | https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85 |
| Kinova Gen3 Menagerie asset | `assets/kinova_gen3/` is derived from MuJoCo Menagerie's Kinova Gen3 model and modified for this project. | BSD 3-Clause License, copyright Kinova inc. | https://github.com/google-deepmind/mujoco_menagerie/tree/main/kinova_gen3 |
The wheel copies the applicable license texts next to its bundled shared libraries; the CMake
install puts them, with `LICENSE`, `NOTICE` and this file, in `share/doc/mj_kdl_wrapper/`:

- `LGPL-2.1.txt` and `KDL_SOURCE.txt` accompany bundled `liborocos-kdl.so*` and `PyKDL*.so`.
- `Apache-2.0.txt` accompanies the installed `libmujoco` and the render and viewer libraries.
- `Robotiq-2F85-BSD-2-Clause.txt` and `Kinova-Gen3-BSD-3-Clause.txt` cover the bundled Robotiq
  2F-85 and Kinova Gen3 assets; only the wheel ships assets, so only the wheel ships these.

See `NOTICE` for attribution notes.
