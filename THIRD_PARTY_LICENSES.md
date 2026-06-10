# Third-Party Licenses

This project depends on and/or vendors the following third-party components.

| Component | Use | License | Source URL |
| --- | --- | --- | --- |
| MuJoCo | Runtime dependency via the pinned `mujoco` Python package; C++ builds link against a matching MuJoCo install. | Apache License 2.0 | https://github.com/google-deepmind/mujoco |
| MuJoCo simulate | `src/simulate_ui/` is derived from MuJoCo's `simulate` sample and modified for this project. | Apache License 2.0 | https://github.com/google-deepmind/mujoco/tree/main/simulate |
| Orocos KDL fork | KDL library and PyKDL bindings built from the secorolab fork when this project owns the KDL build. | GNU Lesser General Public License 2.1 | https://github.com/secorolab/orocos_kinematics_dynamics |
| Robotiq 2F-85 Menagerie asset | `assets/robotiq_2f85/` is derived from MuJoCo Menagerie's Robotiq 2F-85 model and modified for this project. | BSD 2-Clause style license, copyright ROS-Industrial | https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85 |
| MuJoCo Menagerie | Optional robot model assets used by examples and tests; fetched under `third_party/menagerie/` when requested. | Varies by model; see the upstream aggregate and per-model `LICENSE` files. | https://github.com/google-deepmind/mujoco_menagerie |

The packaged wheel and CMake install copy the applicable license texts next to bundled shared libraries:

- `LGPL-2.1.txt` and `KDL_SOURCE.txt` accompany bundled `liborocos-kdl.so*` and `PyKDL*.so`.
- `Apache-2.0.txt` accompanies binaries that include the vendored and modified MuJoCo simulate code.
- `Robotiq-2F85-BSD-2-Clause.txt` covers the bundled Robotiq 2F-85 asset.

See `NOTICE` for attribution notes.
