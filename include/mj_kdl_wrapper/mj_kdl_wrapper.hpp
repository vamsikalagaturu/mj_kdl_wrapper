/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#pragma once

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <kdl/chain.hpp>
#include <kdl/jntarray.hpp>
#include <functional>
#include <string>
#include <vector>

namespace mj_kdl {

/*
 * Log verbosity level.  Each level includes all levels below it:
 *   NONE   - nothing printed.
 *   INFO   - informational messages only (scene/chain construction progress).
 *   WARN   - INFO + recoverable warnings (e.g. fallback to headless mode).
 *   ERROR  - all messages, including errors that cause functions to fail.  Default.
 */
enum class LogLevel { NONE = 0, INFO = 1, WARN = 2, ERROR = 3 };

/* Set the library-wide log verbosity.  Thread-safe for the set call itself;
 * log messages may interleave if called from multiple threads simultaneously. */
void set_log_level(LogLevel level);

/* Return the current log verbosity level. */
LogLevel get_log_level();

/*
 * Placement specification for one robot in a scene.
 * path is interpreted as URDF by build_scene_from_urdfs() and as MJCF by build_scene_from_mjcfs().
 * All joint/body names in MuJoCo are prefixed with `prefix`; must be unique
 * per robot (e.g. "" for arm 0, "r2_" for arm 1).
 */
struct RobotSpec
{
    const char *path     = nullptr;     // URDF or MJCF path (context-dependent)
    const char *prefix   = "";          // name prefix for multi-robot disambiguation
    double      pos[3]   = { 0, 0, 0 }; // world-frame position [m]
    double      euler[3] = { 0, 0, 0 }; // extrinsic XYZ Euler angles [degrees]
};

/*
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
    float  rgba[4]     = { 0.55f, 0.37f, 0.18f, 1.0f }; // wood-ish brown
};

/* Shape type for scene objects. */
enum class Shape { BOX, SPHERE, CYLINDER };

/*
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

/* Full scene description passed to build_scene_from_urdfs(). */
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

/*
 * Joint-space control mode for update().
 *   POSITION - writes jnt_pos_cmd to actuator ctrl inputs.
 *   VELOCITY - writes jnt_vel_cmd to actuator ctrl inputs.
 *   TORQUE   - writes jnt_trq_cmd to qfrc_applied (generalized forces).
 */
enum class CtrlMode { POSITION, VELOCITY, TORQUE };

/*
 * Runtime handle for one KDL-tracked articulation inside a MuJoCo scene.
 * model/data are borrowed (never freed by cleanup()); call destroy_scene() separately.
 *
 * Workflow:
 *   1. Call init_robot_from_urdf() or init_robot_from_mjcf() - populates configuration
 *      and sizes port vectors to n_joints.
 *   2. Each control step: read *_msr ports (updated by update()), fill *_cmd ports,
 *      call update() to apply commands to MuJoCo and read back sensor state.
 */
struct Robot
{
    /* Configuration - set once by init_robot() / init_from_mjcf(). */
    mjModel                               *model    = nullptr;
    mjData                                *data     = nullptr;
    KDL::Chain                             chain;
    int                                    n_joints = 0;
    std::vector<std::string>               joint_names;
    std::vector<std::pair<double, double>> joint_limits;

    /* Ports - read/written each control cycle. */
    CtrlMode            ctrl_mode    = CtrlMode::POSITION;
    bool                paused       = false;
    std::vector<double> jnt_pos_msr; // [rad]   - measured joint positions   (written by update())
    std::vector<double> jnt_vel_msr; // [rad/s] - measured joint velocities  (written by update())
    std::vector<double> jnt_trq_msr; // [Nm]    - bias torques (grav+Cor)    (written by update())
    std::vector<double> jnt_pos_cmd; // [rad]   - position setpoints         (POSITION mode)
    std::vector<double> jnt_vel_cmd; // [rad/s] - velocity setpoints         (VELOCITY mode)
    std::vector<double> jnt_trq_cmd; // [Nm]    - torque commands            (TORQUE mode)

    /* Internal state - populated by init_robot() / init_from_mjcf(). */
    std::vector<int>    kdl_to_mj_qpos; // KDL index -> MuJoCo qpos address
    std::vector<int>    kdl_to_mj_dof;  // KDL index -> MuJoCo dof address
    std::vector<int>    kdl_to_mj_ctrl; // KDL index -> MuJoCo ctrl index (-1 if none)
};

/*
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
    bool        show_joints = true; // toggle joint value overlay (key J)
};

/*
 * Load a MuJoCo MJCF scene directly.
 * @param[out] out_model  Newly allocated MuJoCo model; free via destroy_scene().
 * @param[out] out_data   Newly allocated MuJoCo data; free via destroy_scene().
 * @param[in]  path       Path to the MJCF XML file.
 * @return true on success.
 */
bool load_mjcf(mjModel **out_model, mjData **out_data, const char *path);

/*
 * Add a sky gradient texture and directional light to an existing MJCF file in-place.
 * @param mjcf_path  Path to the MJCF file to patch.
 * @return true on success.
 */
bool patch_mjcf_add_skybox(const char *mjcf_path);

/*
 * Add a checker groundplane texture, material, and floor geom to an existing MJCF
 * file in-place.
 * @param mjcf_path  Path to the MJCF file to patch.
 * @return true on success.
 */
bool patch_mjcf_add_floor(const char *mjcf_path);

/*
 * Append SceneObjects to an existing MJCF file in-place.
 * Equivalent to add_objects_to_spec() for the load_mjcf() / attach_gripper() workflow.
 * Each object becomes a body under worldbody with a freejoint (unless fixed=true).
 * @param mjcf_path  Path to the MJCF file to patch.
 * @param objects    Objects to add.
 * @return true on success.
 */
bool patch_mjcf_add_objects(const char *mjcf_path, const std::vector<SceneObject> &objects);

/*
 * Add contact exclusion pairs to an existing MJCF file in-place.
 * Each entry in exclusions is a (body1, body2) pair; MuJoCo will not generate
 * contacts between them.  Typical use: prevent spurious contacts between an arm
 * link and the gripper bodies after attach_gripper().
 * @param mjcf_path   Path to the MJCF file to patch.
 * @param exclusions  Pairs of body names to exclude from contact.
 * @return true on success.
 */
bool patch_mjcf_contact_exclusions(const char            *mjcf_path,
  const std::vector<std::pair<std::string, std::string>> &exclusions);

/*
 * Save the compiled model to an MJCF XML file for later reloading with load_mjcf().
 * Must be called with the model returned by the most recent build_scene_from_urdfs() or
 * load_mjcf() call  - MuJoCo only retains the last compiled model's XML internally.
 * Typical use: build a combined scene (dual-arm, arm+gripper, ...) once, save it,
 * then reload with load_mjcf() in subsequent runs to skip all build/patch steps.
 * @param model  Model to save; must be the most recently compiled model.
 * @param path   Output path for the MJCF XML file.
 * @return true on success.
 */
bool save_model_xml(const mjModel *model, const char *path);

/*
 * Build KDL chain from a compiled MuJoCo model (no URDF required).
 * Traverses the body tree from base_body to tip_body.
 * @param[out] r          Robot populated with chain, joint_names, joint_limits, index maps.
 * @param[in]  model      Compiled MuJoCo model.
 * @param[in]  data       MuJoCo data pointer.
 * @param[in]  base_body  Name of the chain root body (not included as a segment).
 * @param[in]  tip_body   Name of the chain end body.
 * @param[in]  prefix     Optional body/joint name prefix for multi-robot disambiguation.
 * @return true on success.
 */
bool init_robot_from_mjcf(Robot *r,
  mjModel                       *model,
  mjData                        *data,
  const char                    *base_body,
  const char                    *tip_body,
  const char                    *prefix = "");

/*
 * Gripper attachment spec.
 */
struct GripperSpec
{
    const char *mjcf_path = nullptr;     // gripper MJCF file path
    const char *attach_to = nullptr;     // body name in arm MJCF to attach gripper base to
    double      pos[3]    = { 0, 0, 0 };   // position offset from attach_to body
    double      euler[3]  = { 0, 0, 0 };   // orientation offset, extrinsic XYZ Euler angles [degrees]
    const char *prefix    = "";          // prefix for gripper names (avoids conflicts)

    /* Optional post-attach patches applied inside attach_gripper(). */
    bool                                                add_skybox   = false;
    bool                                                add_floor    = false;
    std::vector<std::pair<std::string, std::string>>    contact_exclusions; // (body1, body2) pairs
};

/*
 * Combine an arm MJCF and a gripper MJCF into a single MJCF file.
 * The gripper's root body is placed as a child of the specified attach_to body.
 * Merges assets, defaults, contacts, tendons, equalities, and actuators.
 * @param[in]  arm_mjcf   Path to the arm MJCF.
 * @param[in]  g          Gripper specification.
 * @param[in]  out_path   Output path for the combined MJCF.
 * @return true on success.
 */
bool attach_gripper(const char *arm_mjcf, const GripperSpec *g, const char *out_path);

/*
 * Merge one or more arm MJCFs into a single world MJCF file.
 * Each arm is placed at the given position/orientation.  All element names (bodies,
 * joints, actuators, geoms, sites) are prefixed with RobotSpec::prefix so
 * multiple instances of the same robot can coexist without name conflicts.
 * Shared assets (meshes, materials, textures) are copied from the first arm
 * only; subsequent arms reuse them.
 * @param[out] out_model  Newly allocated MuJoCo model; caller must free via destroy_scene().
 * @param[out] out_data   Newly allocated MuJoCo data; caller must free via destroy_scene().
 * @param[in]  arms       Array of robot specs; RobotSpec::path is the MJCF file.
 * @param[in]  n_arms     Number of entries in arms[].
 * @param[in]  add_floor  Insert a ground-plane geom.
 * @param[in]  add_skybox Insert a skybox texture.
 * @return true on success.
 */
bool build_scene_from_mjcfs(mjModel **out_model,
  mjData                            **out_data,
  const RobotSpec                    *arms,
  int                                 n_arms,
  bool                                add_floor  = true,
  bool                                add_skybox = true,
  const char                         *out_path   = nullptr);

/*
 * Build a MuJoCo world from one or more URDF robots into a single mjModel/mjData.
 * Also injects any table and objects declared in spec->table and spec->objects.
 * @param[out] out_model  Newly allocated MuJoCo model; caller must free via destroy_scene().
 * @param[out] out_data   Newly allocated MuJoCo data; caller must free via destroy_scene().
 * @param[in]  spec       Scene description: robots, table, objects, timestep, gravity, floor.
 * @return true on success, false on any load or compile error.
 */
bool build_scene_from_urdfs(mjModel **out_model, mjData **out_data, const SceneSpec *spec);

/*
 * Free a model/data pair allocated by build_scene_from_urdfs().
 * @param[in] model  Model to free (may be null).
 * @param[in] data   Data to free (may be null).
 */
void destroy_scene(mjModel *model, mjData *data);

/*
 * Attach a KDL chain to an already-loaded scene (shared model/data  - not owned).
 * Resolves joint names using `prefix` to disambiguate robots in multi-robot scenes.
 * @param[out] r          Robot to populate (chain, joint maps, index maps).
 * @param[in]  model      MuJoCo model from build_scene_from_urdfs(); not freed by cleanup().
 * @param[in]  data       MuJoCo data from build_scene_from_urdfs(); not freed by cleanup().
 * @param[in]  urdf_path  Path to the robot URDF (used to build the KDL chain).
 * @param[in]  base_link  Name of the chain's root link in the URDF.
 * @param[in]  tip_link   Name of the chain's end-effector link in the URDF.
 * @param[in]  prefix     Optional MuJoCo joint-name prefix (default ""); must match
 * RobotSpec::prefix.
 * @return true on success, false if the chain or any joint cannot be found.
 */
bool init_robot_from_urdf(Robot *r,
  mjModel               *model,
  mjData                *data,
  const char            *urdf_path,
  const char            *base_link,
  const char            *tip_link,
  const char            *prefix = "");

/*
 * Open a GLFW window and initialise MuJoCo visualization contexts.
 * Must be called after init_robot() or init_from_mjcf().
 * @param[out] v      Viewer to initialise; must be zero-initialised before call.
 * @param[in]  r      Robot whose model drives the rendering context.
 * @param[in]  title  Window title string.
 * @param[in]  width  Window width in pixels.
 * @param[in]  height Window height in pixels.
 * @return true on success, false if GLFW or MuJoCo context creation fails.
 */
bool init_window(Viewer *v,
  Robot                 *r,
  const char            *title  = "MuJoCo",
  int                    width  = 1280,
  int                    height = 720);

/*
 * Zero all Robot fields.  Does not free model or data; call destroy_scene() for that.
 * @param[in,out] r  Robot to tear down.
 */
void cleanup(Robot *r);

/*
 * Release the GLFW window and MuJoCo visualization contexts owned by v.
 * @param[in,out] v  Viewer to tear down; all pointers set to null afterwards.
 */
void cleanup(Viewer *v);

/*
 * Advance the simulation by one timestep and call mj_forward().
 * @param[in,out] s  Simulation state.
 */
void step(Robot *s);

/*
 * Advance the simulation by n timesteps.
 * @param[in,out] s  Simulation state.
 * @param[in]     n  Number of steps.
 */
void step_n(Robot *s, int n);

/*
 * Reset simulation to the model's keyframe 0 (or default pose).
 * @param[in,out] s  Simulation state.
 */
void reset(Robot *s);

/*
 * Returns true if the viewer window is open and not scheduled for closing.
 * @param[in] v  Viewer created by init_window().
 */
bool is_running(const Viewer *v);

/*
 * Render the current simulation frame to the viewer window.
 * @param[in,out] v  Viewer created by init_window().
 * @param[in]     r  Robot whose model and data are rendered.
 * @return true if the window is still open after rendering.
 */
bool render(Viewer *v, const Robot *r);

/*
 * One control cycle: read MuJoCo into *_msr, then apply *_cmd to MuJoCo.
 * Read step: qpos -> jnt_pos_msr, qvel -> jnt_vel_msr, qfrc_bias -> jnt_trq_msr.
 * Apply step: POSITION -> data->ctrl, VELOCITY -> data->ctrl, TORQUE -> qfrc_applied.
 * Joints with kdl_to_mj_ctrl[i] == -1 are skipped in POSITION/VELOCITY mode.
 */
void update(Robot *r);

/*
 * Write KDL joint positions into MuJoCo qpos (KDL chain order -> MuJoCo addresses).
 * @param[in,out] r            Robot with a valid data pointer.
 * @param[in]     q            Joint positions in KDL chain order; size must equal r->n_joints.
 * @param[in]     call_forward If true (default), calls mj_forward() after writing qpos
 *                             so that body poses and sensor data are updated immediately.
 */
void set_joint_pos(Robot *r, const KDL::JntArray &q, bool call_forward = true);

/*
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

/*
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
 * @param m           MuJoCo model (caller owns; not freed by this function).
 * @param d           MuJoCo data  (caller owns; not freed by this function).
 * @param path        Filename shown in the title bar (pass "" if not applicable).
 * @param physics_cb  Called each physics step; may be nullptr. */
void run_simulate_ui(mjModel *m, mjData *d, const char *path, ControlCb physics_cb = nullptr);

/*
 * Internal spec-building helpers.
 *
 * These are used internally by build_scene_from_urdfs() and configure_spec(), but are
 * exposed here for advanced callers that construct mjSpec objects directly.
 * They are not part of the stable public API and may change between releases.
 */

/*
 * Add a sky gradient texture and overhead directional light to spec.
 * Corresponds to SceneSpec::add_skybox.
 */
void add_skybox_to_spec(mjSpec *spec);

/*
 * Add a checker groundplane texture, material, and floor plane geom to spec.
 * Corresponds to SceneSpec::add_floor.
 */
void add_floor_to_spec(mjSpec *spec);

/*
 * Add a table (tabletop + four legs) to the world body of spec.
 * @param t  Table geometry, size, and colour.
 */
void add_table_to_spec(mjSpec *spec, const TableSpec &t);

/*
 * Add free-floating or fixed rigid bodies to the world body of spec.
 * @param objects  List of objects to add.
 */
void add_objects_to_spec(mjSpec *spec, const std::vector<SceneObject> &objects);

/*
 * Apply all SceneSpec settings to a parsed mjSpec:
 * physics options (timestep, gravity), compiler flags, and scene decorations
 * (skybox, floor, table, objects) according to the flags in sc.
 */
void configure_spec(mjSpec *spec, const SceneSpec *sc);

/*
 * Compile spec into a model and create its data buffer.
 * spec is always deleted (on success and failure).
 * @param[out] out_model  Newly allocated model on success; null on failure.
 * @param[out] out_data   Newly allocated data on success; null on failure.
 * @return true on success.
 */
bool compile_and_make_data(mjSpec *spec, mjModel **out_model, mjData **out_data);

} // namespace mj_kdl
