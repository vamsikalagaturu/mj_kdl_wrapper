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
 * One robot's placement in a multi-robot scene.
 * All joint/body names in MuJoCo are prefixed with `prefix`; must be unique
 * per robot (e.g. "" for arm 0, "r2_" for arm 1).
 */
struct SceneRobot
{
    const char *urdf_path = nullptr;
    const char *prefix    = "";
    double      pos[3]    = { 0, 0, 0 };
    double      euler[3]  = { 0, 0, 0 }; // degrees, extrinsic XYZ
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
enum class ObjShape { BOX, SPHERE, CYLINDER };

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
    ObjShape    shape       = ObjShape::BOX;
    double      size[3]     = { 0.03, 0.03, 0.03 };
    double      pos[3]      = { 0.0, 0.0, 0.0 };
    float       rgba[4]     = { 1.0f, 0.0f, 0.0f, 1.0f };
    bool        fixed       = false;
    double      mass        = 0.1;
    int         condim      = 3;
    double      friction[3] = { 0.5, 0.005, 0.0001 }; // MuJoCo defaults
};

/* Full scene description passed to build_scene(). */
struct SceneSpec
{
    std::vector<SceneRobot>  robots;
    double                   timestep   = 0.002;
    double                   gravity_z  = -9.81;
    bool                     add_floor  = true; // checker groundplane geom
    bool                     add_skybox = true; // gradient sky texture + directional light
    TableSpec                table;
    std::vector<SceneObject> objects;
};

/*
 * Runtime handle for one KDL-tracked articulation inside a MuJoCo scene.
 * Holds the scene model/data reference, the KDL chain with joint index maps.
 * Fields are public.  Do not free model/data directly; use destroy_scene() or cleanup().
 */
struct Robot
{
    mjModel                               *model = nullptr;
    mjData                                *data  = nullptr;
    KDL::Chain                             chain;
    int                                    n_joints = 0;
    std::vector<std::string>               joint_names;
    std::vector<std::pair<double, double>> joint_limits;
    std::vector<int>                       kdl_to_mj_qpos; // KDL index -> MuJoCo qpos address
    std::vector<int>                       kdl_to_mj_dof;  // KDL index -> MuJoCo dof address
    bool _owns_model = true;                               // if true, cleanup() frees model/data
    bool paused      = false; // set true to pause simulation (step() becomes a no-op)
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
 * Save the compiled model to an MJCF XML file for later reloading with load_mjcf().
 * Must be called with the model returned by the most recent build_scene() or
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
 * @param[out] s          Robot populated with chain, joint_names, joint_limits, index maps.
 * @param[in]  model      Compiled MuJoCo model.
 * @param[in]  data       MuJoCo data pointer.
 * @param[in]  base_body  Name of the chain root body (not included as a segment).
 * @param[in]  tip_body   Name of the chain end body.
 * @param[in]  prefix     Optional body/joint name prefix for multi-robot disambiguation.
 * @return true on success.
 */
bool init_from_mjcf(Robot *s,
  mjModel                 *model,
  mjData                  *data,
  const char              *base_body,
  const char              *tip_body,
  const char              *prefix = "");

/*
 * Gripper attachment spec.
 */
struct GripperSpec
{
    const char *mjcf_path = nullptr;        // gripper MJCF file path
    const char *attach_to = nullptr;        // body name in arm MJCF to attach gripper base to
    double      pos[3]    = { 0, 0, 0 };    // position offset from attach_to body
    double      quat[4]   = { 1, 0, 0, 0 }; // orientation offset (wxyz)
    const char *prefix    = "";             // prefix for gripper names (avoids conflicts)
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
 * Placement specification for one arm MJCF in a multi-arm scene.
 * Used with build_scene_from_mjcfs().
 */
struct MjcfArmSpec
{
    const char *mjcf_path = nullptr;     // path to arm (or arm+gripper) MJCF
    const char *prefix    = "";          // prefix applied to all named elements; "" = no prefix
    double      pos[3]    = { 0, 0, 0 }; // placement position in world [m]
    double      euler_z   = 0.0;         // yaw angle in degrees
};

/*
 * Merge one or more arm MJCFs into a single world MJCF file.
 * Each arm is placed at the given position/yaw.  All element names (bodies,
 * joints, actuators, geoms, sites) are prefixed with MjcfArmSpec::prefix so
 * multiple instances of the same robot can coexist without name conflicts.
 * Shared assets (meshes, materials, textures) are copied from the first arm
 * only; subsequent arms reuse them.
 * @param[in]  out_mjcf   Output path for the merged MJCF.
 * @param[in]  arms       Array of arm specs.
 * @param[in]  n_arms     Number of entries in arms[].
 * @param[in]  add_floor  Insert a ground-plane geom.
 * @param[in]  add_skybox Insert a skybox texture.
 * @return true on success.
 */
bool build_scene_from_mjcfs(const char *out_mjcf,
  const MjcfArmSpec                    *arms,
  int                                   n_arms,
  bool                                  add_floor  = true,
  bool                                  add_skybox = true);

/*
 * Build a MuJoCo world from one or more URDF robots into a single mjModel/mjData.
 * Also injects any table and objects declared in spec->table and spec->objects.
 * @param[out] out_model  Newly allocated MuJoCo model; caller must free via destroy_scene().
 * @param[out] out_data   Newly allocated MuJoCo data; caller must free via destroy_scene().
 * @param[in]  spec       Scene description: robots, table, objects, timestep, gravity, floor.
 * @return true on success, false on any load or compile error.
 */
bool build_scene(mjModel **out_model, mjData **out_data, const SceneSpec *spec);

/*
 * Free a model/data pair allocated by build_scene().
 * @param[in] model  Model to free (may be null).
 * @param[in] data   Data to free (may be null).
 */
void destroy_scene(mjModel *model, mjData *data);

/*
 * Attach a KDL chain to an already-loaded scene (shared model/data  - not owned).
 * Resolves joint names using `prefix` to disambiguate robots in multi-robot scenes.
 * @param[out] s          Robot to populate (chain, joint maps, index maps).
 * @param[in]  model      MuJoCo model from build_scene(); not freed by cleanup().
 * @param[in]  data       MuJoCo data from build_scene(); not freed by cleanup().
 * @param[in]  urdf_path  Path to the robot URDF (used to build the KDL chain).
 * @param[in]  base_link  Name of the chain's root link in the URDF.
 * @param[in]  tip_link   Name of the chain's end-effector link in the URDF.
 * @param[in]  prefix     Optional MuJoCo joint-name prefix (default ""); must match
 * SceneRobot::prefix.
 * @return true on success, false if the chain or any joint cannot be found.
 */
bool init_robot(Robot *s,
  mjModel             *model,
  mjData              *data,
  const char          *urdf_path,
  const char          *base_link,
  const char          *tip_link,
  const char          *prefix = "");

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
 * Release model/data if _owns_model and zero all Robot fields.
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
 * Copy MuJoCo qpos into q, reordered to match KDL chain joint order.
 * Resizes q to s->n_joints if needed.
 * @param[in]  s  Simulation state with a valid data pointer.
 * @param[out] q  Joint positions in KDL chain order.
 * @return true on success, false if s->data is null.
 */
bool sync_to_kdl(const Robot *s, KDL::JntArray &q);

/*
 * Write KDL joint positions into MuJoCo qpos (KDL chain order -> MuJoCo addresses).
 * @param[in,out] s  Simulation state.
 * @param[in]     q  Joint positions in KDL chain order; size must equal s->n_joints.
 */
void sync_from_kdl(Robot *s, const KDL::JntArray &q);

/*
 * Apply joint torques by writing into s->data->qfrc_applied (KDL chain order).
 * @param[in,out] s    Simulation state.
 * @param[in]     tau  Torques in KDL chain order; size must equal s->n_joints.
 */
void set_torques(Robot *s, const KDL::JntArray &tau);

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
 * These are used internally by build_scene() and configure_spec(), but are
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
