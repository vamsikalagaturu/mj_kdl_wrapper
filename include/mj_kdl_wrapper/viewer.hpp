/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#pragma once

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <array>
#include <chrono>

namespace mj_kdl {

/**
 * @ingroup grp_viewer
 * A MuJoCo Studio window on an Env, from the mj_kdl_wrapper::viewer component. The caller owns
 * it: open_viewer() opens it, cleanup(Viewer *) or the destructor closes it, and so does
 * cleanup(Env *) of the Env it shows.
 *
 * The window runs on a thread of its own and draws a copy of the Env's state that step()
 * publishes; the Env is only ever read and written on the thread that steps it. Pause, reset,
 * perturbation, edited controls, physics options and recording reach the Env inside step().
 */
struct Viewer
{
    mjvCamera cam{}; // the camera the window opens with; use_camera()/set_free_camera() move it
    /* Real-time factor for pace_realtime(): 1.0 = real time, 0.5 = half speed, 0.0 = uncapped.
     * The ',' and '.' keys adjust it. */
    double realtime_factor = 1.0;
    double fps             = 0.0; // the window's frame rate, read-only, set by step()
    std::chrono::steady_clock::time_point _tick_t{};       // internal: pacing
    void                                 *_impl = nullptr; // internal: open window

    Viewer() = default;
    ~Viewer();
    Viewer(const Viewer &)            = delete;
    Viewer &operator=(const Viewer &) = delete;
};

/**
 * @ingroup grp_viewer
 * Open a Studio window (Filament rendering, Dear ImGui panels) on env. From then on step()
 * honours its pause, reset, perturbation, edited controls and physics options, and records
 * when asked to. Linux (X11, or Wayland through an offscreen context) only.
 * @return an error when there is no display, env is empty, or v is already open.
 */
Status open_viewer(Viewer *v, Env *env, const char *title = "MuJoCo");

/**
 * @ingroup grp_viewer
 * Close the window, if open, and stop a recording it started.
 */
void cleanup(Viewer *v);

/**
 * @ingroup grp_viewer
 * Sleep out this step's share of wall time at the viewer's real-time factor. step() never
 * sleeps; a loop that paces itself reads Viewer::realtime_factor instead. No-op while the
 * viewer is closed.
 */
void pace_realtime(Viewer *v);

/**
 * @ingroup grp_viewer
 * Returns true while the viewer window is open.
 */
bool is_running(const Viewer *v);

/**
 * @ingroup grp_viewer
 * Whether a key is currently held down in the viewer's window, from any thread.
 * Keys the UI consumes for itself (',' and '.' for the speed control) are reported like any
 * other; keys typed into a text field are not.
 * @param[in] v          Viewer, or nullptr.
 * @param[in] imgui_key  An ImGuiKey code from <imgui.h>, e.g. ImGuiKey_UpArrow.
 * @return true while the key is held; false for a nullptr or closed viewer, or a code that
 *         names no key.
 */
bool key_pressed(const Viewer *v, int imgui_key);

/**
 * @ingroup grp_viewer
 * Claim a key for the caller, so the viewer's own bindings (Studio's) never act on it: space
 * pauses, backspace resets, escape restores the free camera, '-' and '=' change the speed, and
 * the letter and digit keys toggle visualization flags and groups. A captured key is still
 * reported by key_pressed().
 * @param[in,out] v          Viewer, or nullptr.
 * @param[in]     imgui_key  An ImGuiKey code from <imgui.h>, e.g. ImGuiKey_Space.
 * @param[in]     capture    true to claim the key, false to give it back.
 */
void capture_key(Viewer *v, int imgui_key, bool capture = true);

/**
 * @ingroup grp_viewer
 * Switch the viewer to a named fixed camera defined in the model.
 * Pass nullptr or an empty string to return to the free camera.
 * @return true if the camera name was found; false if not found (viewer unchanged).
 */
bool use_camera(Viewer *v, const mjModel *model, const char *name);

/**
 * @ingroup grp_viewer
 * Configure the viewer's free orbit camera.
 */
void set_free_camera(
  Viewer                      *v,
  double                       distance,
  double                       azimuth,
  double                       elevation,
  const std::array<double, 3> &lookat
);

} // namespace mj_kdl
