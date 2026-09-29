# Third-Party Licenses

This project depends on and/or vendors the following third-party components.

| Component | Use | License | Source URL |
| --- | --- | --- | --- |
| MuJoCo | Runtime dependency via the pinned `mujoco` Python package; C++ builds link against a matching MuJoCo install. | Apache License 2.0 | https://github.com/google-deepmind/mujoco |
| MuJoCo simulate | `src/simulate_ui/` is derived from MuJoCo's `simulate` sample and modified for this project. | Apache License 2.0 | https://github.com/google-deepmind/mujoco/tree/main/simulate |
| Orocos KDL fork | KDL library and PyKDL bindings built from the secorolab fork when this project owns the KDL build. | GNU Lesser General Public License 2.1 | https://github.com/secorolab/orocos_kinematics_dynamics |
| Robotiq 2F-85 Menagerie asset | `assets/robotiq_2f85/` is derived from MuJoCo Menagerie's Robotiq 2F-85 model and modified for this project. | BSD 2-Clause style license, copyright ROS-Industrial | https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85 |
| Kinova Gen3 Menagerie asset | `assets/kinova_gen3/` is derived from MuJoCo Menagerie's Kinova Gen3 model and modified for this project. | BSD 3-Clause License, copyright Kinova inc. | https://github.com/google-deepmind/mujoco_menagerie/tree/main/kinova_gen3 |
| MuJoCo Menagerie | Optional robot model assets used by examples and tests; fetched into the user cache (`~/.cache/mj_kdl_wrapper/menagerie/`) when requested. | Varies by model; see the upstream aggregate and per-model `LICENSE` files. | https://github.com/google-deepmind/mujoco_menagerie |

The wheel copies the applicable license texts next to its bundled shared libraries; the CMake
install puts them, with `LICENSE`, `NOTICE` and this file, in `share/doc/mj_kdl_wrapper/`:

- `LGPL-2.1.txt` and `KDL_SOURCE.txt` accompany bundled `liborocos-kdl.so*` and `PyKDL*.so`.
- `Apache-2.0.txt` accompanies binaries that include the vendored and modified MuJoCo simulate code.
- `Robotiq-2F85-BSD-2-Clause.txt` and `Kinova-Gen3-BSD-3-Clause.txt` cover the bundled Robotiq
  2F-85 and Kinova Gen3 assets; only the wheel ships assets, so only the wheel ships these.

See `NOTICE` for attribution notes.
