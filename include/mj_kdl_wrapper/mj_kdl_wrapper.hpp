/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#pragma once

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <kdl/chain.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <functional>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace mj_kdl {

/**
 * @ingroup grp_logging
 * Log verbosity level.  Each level includes all levels below it:
 *   NONE   - nothing printed.
 *   INFO   - informational messages only (scene/chain construction progress).
 *   WARN   - INFO + recoverable warnings (e.g. fallback to headless mode).
 *   ERROR  - all messages, including errors that cause functions to fail.  Default.
 */
enum class LogLevel { NONE = 0, INFO = 1, WARN = 2, ERROR = 3 };

/** @ingroup grp_logging
 *  Library-wide log verbosity (inline so one shared instance across all TUs). */
inline LogLevel g_log_level = LogLevel::ERROR;

/** @ingroup grp_logging
 *  Set the library-wide log verbosity. */
inline void set_log_level(LogLevel level) { g_log_level = level; }
/** @ingroup grp_logging
 *  Get the library-wide log verbosity. */
inline LogLevel get_log_level() { return g_log_level; }

/* Logging macros - emit to stderr with colour, file:line, and function name. */
#define MJ_FILENAME_ (::strrchr(__FILE__, '/') ? ::strrchr(__FILE__, '/') + 1 : __FILE__)

#define MJ_LOG_(lvl_enum, color, label, expr)                        \
    do {                                                             \
        if (::mj_kdl::g_log_level >= ::mj_kdl::LogLevel::lvl_enum) { \
            std::ostringstream _mj_oss;                              \
            _mj_oss << expr;                                         \
            std::fprintf(                                            \
              stderr,                                                \
              color "[mj_kdl " label "] %s:%d (%s): %s\033[0m\n",    \
              MJ_FILENAME_,                                          \
              __LINE__,                                              \
              __func__,                                              \
              _mj_oss.str().c_str()                                  \
            );                                                       \
        }                                                            \
    } while (0)

#define LOG_INFO(expr) MJ_LOG_(INFO, "", "INFO ", expr)
#define LOG_WARN(expr) MJ_LOG_(WARN, "\033[33m", "WARN ", expr)
#define LOG_ERROR(expr) MJ_LOG_(ERROR, "\033[31m", "ERROR", expr)

/**
 * @ingroup grp_types
 * One link in an ordered attachment chain for a robot.
 * An attachment is any MJCF body (end effector, mount, FT sensor, tool, additional arm
 * on a mobile base, etc.) attached under a named body in the accumulated robot spec.
 * Attachments are applied in declaration order; attach_to may reference any body present
 * after all prior attachments have been applied.
 */
struct AttachmentSpec
{
    const char *mjcf_path = nullptr;     // MJCF file for this attachment
    const char *attach_to = nullptr;     // body to attach under (in root or prior attachment)
    double      pos[3]    = { 0, 0, 0 }; // position offset [m]
    double      euler[3]  = { 0, 0, 0 }; // extrinsic XYZ Euler offset [degrees]
    const char *prefix    = "";          // element name prefix (avoids name conflicts)

    /* Contact exclusion pairs registered by attach_to_spec(). */
    std::vector<std::pair<std::string, std::string>> contact_exclusions; // (body1, body2) pairs
};

/**
 * @ingroup grp_types
 * One robot in a scene: a root MJCF (arm, mobile base, ...) with an ordered attachment
 * chain and world-frame placement.
 *
 * attachments is applied in order by build_scene() / attach_to_spec(): each entry's
 * attach_to may reference any body in the accumulated spec (root + all prior attachments).
 * This naturally supports: fixed arm, arm+gripper, arm+mount+FT+gripper, mobile base,
 * mobile manipulator (base root, arm as first attachment), and so on.
 *
 * path is the root MJCF passed to build_scene(). prefix must be unique per robot
 * in multi-robot scenes.
 */
struct RobotSpec
{
    const char                 *path = nullptr;         // root MJCF path
    std::vector<AttachmentSpec> attachments;            // ordered attachment chain; empty = none
    const char                 *prefix   = "";          // element name prefix
    double                      pos[3]   = { 0, 0, 0 }; // world-frame position [m]
    double                      euler[3] = { 0, 0, 0 }; // extrinsic XYZ Euler angles [degrees]
};

/**
 * @ingroup grp_types
 * Optional table to include in the scene.
 * pos[2] is the table TOP SURFACE height (where robots and objects rest).
 * Legs extend from the bottom of the tabletop panel down to z = 0.
 */
struct TableSpec
{
    bool   enabled     = false;
    double pos[3]      = { 0.0, 0.0, 0.7 };             // (x, y, surface_z)
    double top_size[2] = { 0.6, 0.4 };                  // tabletop half-extents in x, y
    double thickness   = 0.04;                          // full thickness of tabletop panel
    double leg_radius  = 0.025;                         // leg cylinder radius
    float  rgba[4]     = { 0.45f, 0.28f, 0.12f, 1.0f }; // tabletop colour (warm walnut)
    float  leg_rgba[4] = { 0.35f, 0.35f, 0.35f, 1.0f }; // leg colour (dark gray metal)
};

/** @ingroup grp_types
 *  Shape type for scene objects. */
enum class Shape { BOX, SPHERE, CYLINDER };

/**
 * @ingroup grp_types
 * A free-floating or fixed rigid body to place in the scene.
 *
 * size:
 *   BOX       - half-extents (x, y, z)
 *   SPHERE    - {radius, 0, 0}
 *   CYLINDER  - {radius, half-length, 0}
 *
 * pos:
 *   World-frame position. To rest on the table set
 *   pos[2] = table.pos[2] + half-height (e.g. box: pos[2] = surface_z + size[2]).
 *
 * fixed:
 *   If true the body is welded to the world (no freejoint); useful for
 *   static obstacles or fixtures on the table.
 */
struct SceneObject
{
    std::string name;
    Shape       shape       = Shape::BOX;
    double      size[3]     = { 0.03, 0.03, 0.03 };
    double      pos[3]      = { 0.0, 0.0, 0.0 };
    float       rgba[4]     = { 1.0f, 0.0f, 0.0f, 1.0f };
    bool        fixed       = false;
    double      mass        = 0.1;
    int         condim      = 3;
    double      friction[3] = { 0.5, 0.005, 0.0001 }; // MuJoCo defaults
};

/** @ingroup grp_types
 *  Full scene description passed to build_scene(). */
struct SceneSpec
{
    std::vector<RobotSpec>   robots;
    double                   timestep   = 0.002;
    double                   gravity_z  = -9.81;
    bool                     add_floor  = true; // checker groundplane geom
    bool                     add_skybox = true; // gradient sky texture + directional light
    TableSpec                table;
    std::vector<SceneObject> objects;
};

/**
 * @ingroup grp_types
 * Optional tool/end-effector description used while building the KDL chain.
 *
 * tool_body names the root of the attached tool subtree whose mass/inertia is
 * lumped into the arm dynamics.  tcp_site names an authored MuJoCo site that
 * becomes the KDL terminal frame for FK/IK (takes priority when set).  When
 * the model has no suitable site, tcp_frame provides an equivalent manual
 * transform expressed in the tip body's local frame.
 * For the prefixed Robotiq 2F-85 this is typically {"g_base", "g_pinch"}.
 */
struct ToolFrameSpec
{
    const char *tool_body = nullptr;
    const char *tcp_site  = nullptr;                 // MuJoCo site name (takes priority)
    KDL::Frame  tcp_frame = KDL::Frame::Identity(); // manual TCP in tip frame (fallback)
};

/**
 * @ingroup grp_types
 * Joint-space control mode for update().
 *   POSITION - writes jnt_pos_cmd to actuator ctrl inputs.
 *   TORQUE   - writes jnt_trq_cmd to qfrc_applied (generalized forces).
 */
enum class CtrlMode { POSITION, TORQUE };

/**
 * @ingroup grp_types
 * Runtime handle for one KDL-tracked articulation inside a MuJoCo scene.
 * model/data are borrowed (never freed by cleanup()); call destroy_scene() separately.
 *
 * Workflow:
 *   1. Call init_robot_from_mjcf() - populates configuration and sizes port vectors to n_joints.
 *   2. Each control step: read *_msr ports (updated by update()), fill *_cmd ports,
 *      call update() to apply commands to MuJoCo and read back sensor state.
 */
struct Robot
{
    /* Configuration - set once by init_robot() / init_from_mjcf(). */
    mjModel                               *model = nullptr;
    mjData                                *data  = nullptr;
    KDL::Chain                             chain;
    KDL::Frame                             tip_T_tcp     = KDL::Frame::Identity();
    bool                                   has_tcp_frame = false;
    std::string                            tcp_site;
    int                                    n_joints = 0;
    std::vector<std::string>               joint_names;
    std::vector<std::pair<double, double>> joint_limits;

    /* Ports - read/written each control cycle. */
    CtrlMode            ctrl_mode = CtrlMode::POSITION;
    bool                paused    = false;
    /* Optional callback invoked by step() whenever the Simulate UI Reset button
     * is pressed.  Use this to restore scene-specific state (object poses,
     * gripper position, etc.) that mj_resetData does not cover.
     * Called after mj_resetData + mj_forward, before the next mj_step.
     * The callback receives the model and data pointers for convenience. */
    std::function<void(mjModel *, mjData *)> on_reset;
    std::vector<double> jnt_pos_msr; // [rad]   - measured joint positions   (written by update())
    std::vector<double> jnt_vel_msr; // [rad/s] - measured joint velocities  (written by update())
    std::vector<double> jnt_trq_msr; // [Nm]    - actuator output torques    (written by update())
    std::vector<double> jnt_pos_cmd; // [rad] - position setpoints  (POSITION mode)
    std::vector<double> jnt_trq_cmd; // [Nm]  - torque commands     (TORQUE mode)

    /* Internal state - populated by init_robot() / init_from_mjcf(). */
    std::vector<int> kdl_to_mj_qpos; // KDL index -> MuJoCo qpos address
    std::vector<int> kdl_to_mj_dof;  // KDL index -> MuJoCo dof address
    std::vector<int> kdl_to_mj_ctrl; // KDL index -> MuJoCo ctrl index (-1 if none)
};

/**
 * @ingroup grp_viewer
 * GLFW window and MuJoCo visualization state for the manual render loop.
 * Created by init_window(); freed by cleanup(Viewer *).
 */
struct Viewer
{
    GLFWwindow *window = nullptr;
    mjvScene    scn{};
    mjvCamera   cam{};
    mjvOption   opt{};
    mjvPerturb  pert{};
    mjrContext  con{};
    /* Real-time factor controlling simulation speed in tick().
     * 1.0 = real-time (default), 2.0 = 2x faster, 0.5 = half speed, 0.0 = max speed.
     * In GUI mode the Simulate UI -/= keys adjust this value.
     * 0.0 means run as fast as possible (no sleep).
     * Note: the Simulate UI display caps at 100% (1.0x); faster values show as 100%. */
    double      realtime_factor = 1.0;
    /* internal: real-time pacing state used by tick(). */
    std::chrono::steady_clock::time_point _tick_t{};
    /* internal: non-null when init_window_sim() is used; holds SimUiState*. */
    void *_sim_ui = nullptr;
};

/**
 * @ingroup grp_recorder
 * Standard output resolution presets for init_video_recorder().
 * Each maps to a 16:9 frame size at the named quality level.
 */
enum class VideoResolution {
    R360p  = 360,  // 640  x 360
    R480p  = 480,  // 854  x 480
    R720p  = 720,  // 1280 x 720
    R1080p = 1080, // 1920 x 1080
    R2K    = 1440, // 2560 x 1440
    R4K    = 2160, // 3840 x 2160
};

/**
 * @ingroup grp_recorder
 * Headless video recorder.  Renders frames to an EGL offscreen buffer and
 * pipes raw RGB data to an ffmpeg process, producing an H.264 MP4 without a
 * display server or GLFW window.
 *
 * Requirements: EGL (libegl-dev) and ffmpeg available in PATH.
 *
 * Typical usage:
 *
 *   VideoRecorder vr;
 *   init_video_recorder(&vr, model, "sim.mp4", VideoResolution::R1080p);
 *   vr.cam.azimuth = 135;  vr.cam.elevation = -20;  vr.cam.distance = 2.5;
 *
 *   for (int i = 0; i < steps; ++i) {
 *       mj_step(model, data);
 *       record_frame(&vr, model, data);
 *   }
 *
 *   cleanup(&vr);
 */
struct VideoRecorder
{
    mjvCamera cam{};           // camera configuration; modify freely between frames
    mjvOption opt{};           // rendering options; modify freely between frames
    void     *_impl = nullptr; // opaque EGL + ffmpeg state
};

/**
 * @ingroup grp_scene
 * Save the compiled model to an MJCF XML file for later reloading via build_scene().
 * Must be called with the model returned by the most recent build_scene() call -
 * MuJoCo only retains the last compiled model's XML internally.
 * Typical use: build a combined scene (dual-arm, arm+gripper, ...) once, save it,
 * then reload via build_scene() in subsequent runs to skip all build steps.
 * @param model  Model to save; must be the most recently compiled model.
 * @param path   Output path for the MJCF XML file.
 * @return true on success.
 */
bool save_model_xml(const mjModel *model, const char *path);

/**
 * @ingroup grp_robot
 * Build KDL chain from a compiled MuJoCo model (no URDF required).
 * Traverses the body tree from base_body to tip_body.
 *
 * When tool_body is provided, all bodies in the subtree rooted at tool_body are
 * collected and their mass/inertia is lumped into a single fixed KDL segment
 * appended to the chain.  This makes KDL dynamics (ChainDynParam::JntToGravity,
 * JntToMass, etc.) account for the tool's full inertia without adding any
 * controllable joints.  n_joints remains equal to the number of hinge/slide
 * joints between base_body and tip_body.
 *
 * @param[out] r          Robot populated with chain, joint_names, joint_limits, index maps.
 * @param[in]  model      Compiled MuJoCo model.
 * @param[in]  data       MuJoCo data pointer (mj_forward is called internally if tool_body is set).
 * @param[in]  base_body  Name of the chain root body (not included as a segment).
 * @param[in]  tip_body   Name of the chain end body (last controllable link).
 * @param[in]  prefix     Optional body/joint name prefix for multi-robot disambiguation.
 * @param[in]  tool_body  Optional root of the tool subtree whose inertia is appended as a
 *                        fixed segment.  Pass the gripper base body (e.g. "g_base") so that
 *                        KDL dynamics include the full gripper mass.
 * @return true on success.
 * @deprecated Prefer the ToolFrameSpec overload which also accepts tcp_site/tcp_frame.
 */
[[deprecated("use init_robot_from_mjcf(..., const ToolFrameSpec *tool) to also set tcp_site")]]
bool init_robot_from_mjcf(
  Robot      *r,
  mjModel    *model,
  mjData     *data,
  const char *base_body,
  const char *tip_body,
  const char *prefix,
  const char *tool_body
);

/**
 * @ingroup grp_robot
 * Build KDL chain from a compiled MuJoCo model and optional tool/TCP metadata.
 *
 * If tool->tcp_site is set, that authored site becomes the KDL terminal frame
 * for FK/IK.  The joint count and MuJoCo joint/actuator maps still cover only
 * the controllable joints from base_body to tip_body.
 * Pass tool = nullptr (the default) for an arm with no attached tool.
 */
bool init_robot_from_mjcf(
  Robot               *r,
  mjModel             *model,
  mjData              *data,
  const char          *base_body,
  const char          *tip_body,
  const char          *prefix = "",
  const ToolFrameSpec *tool   = nullptr
);

/**
 * @ingroup grp_scene
 * Apply one attachment to an arm spec using the MuJoCo spec API (mjs_attach).
 * Parses a->mjcf_path, attaches its first root body under a->attach_to with the given
 * pos/euler offset, prefixes all element names with a->prefix, and registers contact
 * exclusions via mjs_addExclude.  Can be called repeatedly to build a chain: each
 * subsequent a->attach_to may reference any body added by prior calls.
 * @param[in,out] robot_spec  Accumulated robot spec to attach into.
 * @param[in]     a           Attachment; a->mjcf_path must not be null.
 * @return true on success.
 */
bool attach_to_spec(mjSpec *robot_spec, const AttachmentSpec *a);

/**
 * @ingroup grp_scene
 * Build a MuJoCo scene from one or more robots using the MuJoCo spec API.
 * This is the primary scene-building function.
 *
 * For each RobotSpec: mj_parseXML loads the root MJCF, then attach_to_spec() applies
 * each entry in RobotSpec::attachments in order (mount, sensor, gripper, etc.),
 * and mjs_attach places the complete robot spec at the given position.  A single
 * mj_compile produces the final model -- no intermediate XML files are written.
 *
 * @param[out] out_model  Newly allocated MuJoCo model; caller frees via destroy_scene().
 * @param[out] out_data   Newly allocated MuJoCo data; caller frees via destroy_scene().
 * @param[in]  spec       Scene description: robots (with attachment chains), table,
 *                        objects, timestep, gravity, floor, skybox.
 * @return true on success.
 */
bool build_scene(mjModel **out_model, mjData **out_data, const SceneSpec *spec);

/**
 * @ingroup grp_scene
 * Free a model/data pair allocated by any scene-building function.
 * @param[in] model  Model to free (may be null).
 * @param[in] data   Data to free (may be null).
 */
void destroy_scene(mjModel *model, mjData *data);


/**
 * @ingroup grp_viewer
 * Open a GLFW window and initialise MuJoCo visualization contexts.
 * Must be called after init_robot() or init_from_mjcf().
 * @param[out] v      Viewer to initialise; must be zero-initialised before call.
 * @param[in]  r      Robot whose model drives the rendering context.
 * @param[in]  title  Window title string.
 * @param[in]  width  Window width in pixels.
 * @param[in]  height Window height in pixels.
 * @return true on success, false if GLFW or MuJoCo context creation fails.
 */
bool init_window(
  Viewer     *v,
  Robot      *r,
  const char *title  = "MuJoCo",
  int         width  = 1280,
  int         height = 720
);

/**
 * @ingroup grp_viewer
 * Open the full MuJoCo simulate UI (panels, physics controls, joint viewer)
 * in a background render thread, then return so the caller can drive the
 * physics loop with tick().
 *
 * Use this instead of init_window() when you want the simulate UI panels
 * alongside a user-owned loop.  tick() automatically acquires the render
 * mutex, steps physics, and handles pause / perturbation / speed controls.
 *
 * Note: the render thread owns the GLFW window; on Linux (X11 / Wayland)
 * this works correctly.  Not supported on macOS.
 *
 * @param[out] v      Viewer to initialise; freed by cleanup(Viewer *).
 * @param[in]  r      Robot to simulate.  r is registered globally; pass the same
 *                    Robot to every subsequent step() call so that keyboard and
 *                    mouse perturbation callbacks operate on the correct model.
 *                    Only one (Viewer, Robot) pair may be active at a time.
 * @param[in]  title  Label shown in the window title bar (default "MuJoCo").
 * @return true on success.
 */
bool init_window_sim(Viewer *v, Robot *r, const char *title = "MuJoCo");

/**
 * @ingroup grp_robot
 * Zero all Robot fields.  Does not free model or data; call destroy_scene() for that.
 * @param[in,out] r  Robot to tear down.
 */
void cleanup(Robot *r);

/**
 * @ingroup grp_viewer
 * Release the GLFW window and MuJoCo visualization contexts owned by v.
 * @param[in,out] v  Viewer to tear down; all pointers set to null afterwards.
 */
void cleanup(Viewer *v);

/**
 * @ingroup grp_recorder
 * Initialise a headless EGL video recorder.
 * Creates an EGL context, an offscreen render target, and launches an ffmpeg
 * process (H.264/MP4) via a pipe.  The MuJoCo model is used to size the scene
 * and initialise the rendering context; it must remain valid until cleanup().
 *
 * @param vr        VideoRecorder to initialise; freed by cleanup(VideoRecorder*).
 * @param model     MuJoCo model for the rendering context.
 * @param out_path  Output MP4 path (e.g. "sim.mp4").
 * @param width     Frame width in pixels (default 1280).
 * @param height    Frame height in pixels (default 720).
 * @param fps       Playback frame rate (default 60).
 * @return true on success; false if EGL init or ffmpeg launch fails.
 */
bool init_video_recorder(
  VideoRecorder *vr,
  mjModel       *model,
  const char    *out_path,
  int            width  = 1280,
  int            height = 720,
  int            fps    = 60
);

/**
 * @ingroup grp_recorder
 * Convenience overload: initialise a VideoRecorder using a named resolution preset.
 * Frame width is derived from the preset at 16:9 aspect ratio.
 *
 * @param vr         VideoRecorder to initialise.
 * @param model      MuJoCo model.
 * @param out_path   Output MP4 path.
 * @param resolution VideoResolution preset (e.g. VideoResolution::R1080p).
 * @param fps        Playback frame rate (default 60).
 * @return true on success.
 */
bool init_video_recorder(
  VideoRecorder  *vr,
  mjModel        *model,
  const char     *out_path,
  VideoResolution resolution,
  int             fps = 60
);

/**
 * @ingroup grp_recorder
 * Render the current simulation state and write one frame to the video stream.
 * Call mj_step() (or equivalent) before each record_frame() call.
 *
 * @param vr    VideoRecorder initialised by init_video_recorder().
 * @param model MuJoCo model.
 * @param data  MuJoCo data (current state).
 * @return true on success; false on render or pipe write error.
 */
bool record_frame(VideoRecorder *vr, mjModel *model, mjData *data);

/**
 * @ingroup grp_recorder
 * Flush the ffmpeg pipe, finalise the MP4, and release all EGL resources.
 * After this call vr->_impl is null and the VideoRecorder may be discarded.
 *
 * @param vr  VideoRecorder to tear down.
 */
void cleanup(VideoRecorder *vr);

/**
 * @ingroup grp_robot
 * Advance one physics timestep.
 *
 * Headless (no viewer active): calls mj_step() and returns true.
 * GUI (init_window_sim() was called): advances physics, renders, syncs to real
 * time, and polls GLFW events -- exactly what tick() used to do.  Returns false
 * once the user closes the window.
 *
 * This replaces the old headless/GUI split:
 *
 *   // Before:
 *   if (headless) { mj_kdl::step(&r); }
 *   else if (!mj_kdl::tick(&v, m, d)) break;
 *
 *   // After:
 *   if (!mj_kdl::step(&r)) break;
 *
 * Coupling note: in GUI mode the keyboard and mouse perturbation callbacks
 * operate on the Robot registered via init_window_sim().  Always pass the
 * same Robot to both init_window_sim() and step(); passing a different Robot
 * causes perturbation forces to be applied to the wrong model.
 *
 * @param[in,out] s  Simulation state; must be the Robot passed to init_window_sim().
 * @return true while the window is open (or always true in headless mode).
 */
bool step(Robot *s);

/**
 * @ingroup grp_robot
 * Advance the simulation by n timesteps.
 * Returns false immediately if the viewer window is closed mid-sequence.
 * @param[in,out] s  Simulation state.
 * @param[in]     n  Number of steps.
 */
bool step_n(Robot *s, int n);

/**
 * @ingroup grp_viewer
 * Model/data overload of step() for multi-robot or no-robot GUI loops.
 * Equivalent to the former tick(Viewer*, mjModel*, mjData*).
 * @param[in,out] v  Viewer initialised by init_window() or init_window_sim().
 * @param[in]     m  Shared MuJoCo model.
 * @param[in]     d  Shared MuJoCo data.
 * @return true while the window is open; false once the user closes it.
 */
bool step(Viewer *v, mjModel *m, mjData *d);

/**
 * @ingroup grp_robot
 * Reset simulation to keyframe 0 if the model has one, otherwise to the
 * model default pose.  Syncs jnt_pos_cmd to the reset qpos and zeroes
 * jnt_trq_cmd so the first update() after reset applies no stale commands.
 * Invokes on_reset if set, matching the behaviour of the Simulate UI Reset
 * button so headless and GUI reset paths are consistent.
 * @param[in,out] s  Simulation state.
 */
void reset(Robot *s);

/**
 * @ingroup grp_viewer
 * Returns true if the viewer window is open and not scheduled for closing.
 * @param[in] v  Viewer created by init_window().
 */
bool is_running(const Viewer *v);

/**
 * @ingroup grp_viewer
 * Render the current simulation frame to the viewer window.
 * @param[in,out] v  Viewer created by init_window().
 * @param[in]     r  Robot whose model and data are rendered.
 * @return true if the window is still open after rendering.
 */
bool render(Viewer *v, const Robot *r);

/**
 * @ingroup grp_viewer
 * Render the current simulation frame to the viewer window.
 * Model/data overload -- use when no single Robot owns the scene (e.g. multi-robot).
 * @param[in,out] v  Viewer created by init_window().
 * @param[in]     m  MuJoCo model.
 * @param[in]     d  MuJoCo data.
 * @return true if the window is still open after rendering.
 */
bool render(Viewer *v, mjModel *m, mjData *d);

/** @deprecated Use step(Robot *) instead. */
[[deprecated("use step(Robot *)")]] bool tick(Viewer *v, Robot *r);
/** @deprecated Use step(Viewer *, mjModel *, mjData *) instead. */
[[deprecated("use step(Viewer *, mjModel *, mjData *)")]] bool tick(Viewer *v, mjModel *m, mjData *d);

/**
 * @ingroup grp_robot
 * One control cycle: read MuJoCo into *_msr, then apply *_cmd to MuJoCo.
 * Read step: qpos -> jnt_pos_msr, qvel -> jnt_vel_msr, qfrc_actuator -> jnt_trq_msr.
 * Apply step: POSITION -> data->ctrl,
 *             TORQUE   -> qfrc_applied; also sets ctrl = qpos to neutralize
 *             position actuators (zeroes kp*(ctrl-qpos) restoring force).
 * Joints with kdl_to_mj_ctrl[i] == -1 are skipped for ctrl writes.
 */
void update(Robot *r);

/**
 * @ingroup grp_robot
 * Write KDL joint positions into MuJoCo qpos (KDL chain order -> MuJoCo addresses).
 * @param[in,out] r            Robot with a valid data pointer.
 * @param[in]     q            Joint positions in KDL chain order; size must equal r->n_joints.
 * @param[in]     call_forward If true (default), calls mj_forward() after writing qpos
 *                             so that body poses and sensor data are updated immediately.
 */
void set_joint_pos(Robot *r, const KDL::JntArray &q, bool call_forward = true);

/**
 * @ingroup grp_scene
 * Add an object to the scene by appending it to spec->objects and rebuilding
 * the model. The old model/data are freed; new ones replace them.
 * Any Robot handles sharing the old model/data become stale  - call init_robot()
 * again on the new model/data after this call.
 * @param[in,out] model  Current model pointer; updated to new model on success.
 * @param[in,out] data   Current data pointer; updated to new data on success.
 * @param[in,out] spec   Scene spec; obj is appended to spec->objects.
 * @param[in]     obj    Object to add.
 * @return true on success; model/data and spec->objects unchanged on failure.
 */
bool scene_add_object(mjModel **model, mjData **data, SceneSpec *spec, const SceneObject &obj);

/**
 * @ingroup grp_scene
 * Remove a named object from the scene by erasing it from spec->objects and
 * rebuilding the model. The old model/data are freed; new ones replace them.
 * Any Robot handles sharing the old model/data become stale  - call init_robot()
 * again on the new model/data after this call.
 * @param[in,out] model  Current model pointer; updated to new model on success.
 * @param[in,out] data   Current data pointer; updated to new data on success.
 * @param[in,out] spec   Scene spec; named object removed from spec->objects.
 * @param[in]     name   Name of the object to remove.
 * @return true on success; false if name not found or rebuild fails.
 */
bool scene_remove_object(mjModel **model, mjData **data, SceneSpec *spec, const std::string &name);

/* Control callback: called each physics step while the simulation is running.
 * Apply custom forces/torques (ctrl, qfrc_applied) here before mj_step().
 * Mouse perturbation forces are applied automatically by run_simulate_ui(). */
using ControlCb = std::function<void(mjModel *m, mjData *d)>;

/* Run the MuJoCo simulate UI with a real-time physics loop (mirrors the
 * PhysicsLoop pattern from mujoco/simulate/main.cc).
 * Blocks until the window is closed.
 *
 * Deprecated: prefer init_window_sim() + tick() so examples and applications own
 * the control loop and can handle reset-aware state explicitly.
 *
 * @param m           MuJoCo model (caller owns; not freed by this function).
 * @param d           MuJoCo data  (caller owns; not freed by this function).
 * @param path        Filename shown in the title bar (pass "" if not applicable).
 * @param physics_cb  Called each physics step; may be nullptr. */
[[deprecated("prefer init_window_sim() + tick()")]] void
  run_simulate_ui(mjModel *m, mjData *d, const char *path, ControlCb physics_cb = nullptr);

/**
 * Internal spec-building helpers.
 *
 * These are used internally by build_scene() and configure_spec(), but are
 * exposed here for advanced callers that construct mjSpec objects directly.
 * They are not part of the stable public API and may change between releases.
 */

/**
 * @ingroup grp_advanced
 * Add a sky gradient texture and overhead directional light to spec.
 * Corresponds to SceneSpec::add_skybox.
 */
void add_skybox_to_spec(mjSpec *spec);

/**
 * @ingroup grp_advanced
 * Add a checker groundplane texture, material, and floor plane geom to spec.
 * Corresponds to SceneSpec::add_floor.
 */
void add_floor_to_spec(mjSpec *spec);

/**
 * @ingroup grp_advanced
 * Add a table (tabletop + four legs) to the world body of spec.
 * @param spec  MuJoCo spec to modify.
 * @param t     Table geometry, size, and colour.
 */
void add_table_to_spec(mjSpec *spec, const TableSpec &t);

/**
 * @ingroup grp_advanced
 * Add free-floating or fixed rigid bodies to the world body of spec.
 * @param spec     MuJoCo spec to modify.
 * @param objects  List of objects to add.
 */
void add_objects_to_spec(mjSpec *spec, const std::vector<SceneObject> &objects);

/**
 * @ingroup grp_advanced
 * Apply all SceneSpec settings to a parsed mjSpec:
 * physics options (timestep, gravity), compiler flags, and scene decorations
 * (skybox, floor, table, objects) according to the flags in sc.
 */
void configure_spec(mjSpec *spec, const SceneSpec *sc);

/**
 * @ingroup grp_advanced
 * Compile spec into a model and create its data buffer.
 * spec is always deleted (on success and failure).
 * @param[in]  spec       MuJoCo spec to compile; always freed by this call.
 * @param[out] out_model  Newly allocated model on success; null on failure.
 * @param[out] out_data   Newly allocated data on success; null on failure.
 * @return true on success.
 */
bool compile_and_make_data(mjSpec *spec, mjModel **out_model, mjData **out_data);

/**
 * @ingroup grp_advanced
 * Load MuJoCo decoder plugins (STL, OBJ, ...) once at first use.
 * Required for external mesh decoder plugin libraries.
 * Called automatically by all scene-building functions; call explicitly only
 * when building a scene via raw mjSpec APIs without going through the library.
 */
void ensure_plugins_loaded();

} // namespace mj_kdl
