#pragma once

#include <functional>
#include <mujoco/mujoco.h>

namespace mj_kdl {

/// Control callback: called each physics step while the simulation is running.
/// Apply custom forces/torques (ctrl, qfrc_applied) here before mj_step().
/// Mouse perturbation forces are applied automatically by run_simulate_ui().
using ControlCb = std::function<void(mjModel* m, mjData* d)>;

/// Run the MuJoCo simulate UI with a real-time physics loop (mirrors the
/// PhysicsLoop pattern from mujoco/simulate/main.cc).
/// Blocks until the window is closed.
///
/// @param m           MuJoCo model (caller owns; not freed by this function).
/// @param d           MuJoCo data  (caller owns; not freed by this function).
/// @param path        Filename shown in the title bar (pass "" if not applicable).
/// @param physics_cb  Called each physics step; may be nullptr.
void run_simulate_ui(mjModel* m, mjData* d, const char* path,
                     ControlCb physics_cb = nullptr);

} // namespace mj_kdl
