#pragma once

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <kdl/chain.hpp>
#include <kdl/jntarray.hpp>
#include <string>
#include <vector>

namespace mj_kdl {

/* Single-robot convenience config for init(). */
struct Config {
    const char* urdf_path  = nullptr;
    const char* base_link  = nullptr;
    const char* tip_link   = nullptr;
    const char* robot_name = "robot";
    int         win_width  = 1280;
    int         win_height = 720;
    const char* win_title  = "MuJoCo";
    double      timestep   = 0.002;
    double      gravity_z  = -9.81;
    bool        add_floor  = true;
    bool        headless   = false;
};

/*
 * One robot's placement in a multi-robot scene.
 * All joint/body names in MuJoCo are prefixed with `prefix`; must be unique
 * per robot (e.g. "" for arm 0, "r2_" for arm 1).
 */
struct SceneRobot {
    const char* urdf_path = nullptr;
    const char* prefix    = "";
    double      pos[3]    = {0, 0, 0};
    double      euler[3]  = {0, 0, 0}; // degrees, extrinsic XYZ
};

/*
 * Optional table to include in the scene.
 * pos[2] is the table TOP SURFACE height (where robots and objects rest).
 * Legs extend from the bottom of the tabletop panel down to z = 0.
 */
struct TableSpec {
    bool   enabled      = false;
    double pos[3]       = {0.0, 0.0, 0.7};          // (x, y, surface_z)
    double top_size[2]  = {0.6, 0.4};                // tabletop half-extents in x, y
    double thickness    = 0.04;                       // full thickness of tabletop panel
    double leg_radius   = 0.025;                      // leg cylinder radius
    float  rgba[4]      = {0.55f, 0.37f, 0.18f, 1.0f}; // wood-ish brown
};

/* Shape type for scene objects. */
enum class ObjShape { BOX, SPHERE, CYLINDER };

/*
 * A free-floating or fixed rigid body to place in the scene.
 *
 * size:
 *   BOX      — half-extents (x, y, z)
 *   SPHERE   — {radius, 0, 0}
 *   CYLINDER — {radius, half-length, 0}
 *
 * pos:
 *   World-frame position. To rest on the table set
 *   pos[2] = table.pos[2] + half-height (e.g. box: pos[2] = surface_z + size[2]).
 *
 * fixed:
 *   If true the body is welded to the world (no freejoint); useful for
 *   static obstacles or fixtures on the table.
 */
struct SceneObject {
    std::string name;
    ObjShape    shape    = ObjShape::BOX;
    double      size[3]  = {0.03, 0.03, 0.03};
    double      pos[3]   = {0.0,  0.0,  0.0};
    float       rgba[4]  = {1.0f, 0.0f, 0.0f, 1.0f};
    bool        fixed    = false;
};

/* Full scene description passed to build_scene(). */
struct SceneSpec {
    std::vector<SceneRobot>  robots;
    double                   timestep  = 0.002;
    double                   gravity_z = -9.81;
    bool                     add_floor = true;
    TableSpec                table;
    std::vector<SceneObject> objects;
};

/*
 * All runtime state for one articulation. Fields are public.
 * Do not free model/data directly; use destroy_scene() or cleanup().
 */
struct State {
    mjModel* model    = nullptr;
    mjData*  data     = nullptr;
    KDL::Chain chain;
    int        n_joints = 0;
    std::vector<std::string>              joint_names;
    std::vector<std::pair<double,double>> joint_limits;
    std::vector<int> kdl_to_mj_qpos; // KDL index → MuJoCo qpos address
    std::vector<int> kdl_to_mj_dof;  // KDL index → MuJoCo dof address
    GLFWwindow* window = nullptr;
    mjvScene    scn    {};
    mjvCamera   cam    {};
    mjvOption   opt    {};
    mjvPerturb  pert   {};
    mjrContext  con    {};
    bool _owns_model  = true;  // if true, cleanup() frees model/data
    bool paused       = false; // set true to pause simulation (step() becomes a no-op)
    bool show_joints  = true;  // show joint value overlay in the viewer
};

/*
 * Load a MuJoCo MJCF scene directly.
 * @param[out] out_model  Newly allocated MuJoCo model; free via destroy_scene().
 * @param[out] out_data   Newly allocated MuJoCo data; free via destroy_scene().
 * @param[in]  path       Path to the MJCF XML file.
 * @return true on success.
 */
bool load_mjcf(mjModel** out_model, mjData** out_data, const char* path);

/*
 * Build KDL chain from a compiled MuJoCo model (no URDF required).
 * Traverses the body tree from base_body to tip_body.
 * @param[out] s          State populated with chain, joint_names, joint_limits, index maps.
 * @param[in]  model      Compiled MuJoCo model.
 * @param[in]  data       MuJoCo data pointer.
 * @param[in]  base_body  Name of the chain root body (not included as a segment).
 * @param[in]  tip_body   Name of the chain end body.
 * @param[in]  prefix     Optional body/joint name prefix for multi-robot disambiguation.
 * @return true on success.
 */
bool init_from_mjcf(State* s, mjModel* model, mjData* data,
                    const char* base_body, const char* tip_body,
                    const char* prefix = "");

/*
 * Gripper attachment spec.
 */
struct GripperSpec {
    const char* mjcf_path  = nullptr; // gripper MJCF file path
    const char* attach_to  = nullptr; // body name in arm MJCF to attach gripper base to
    double      pos[3]     = {0, 0, 0};       // position offset from attach_to body
    double      quat[4]    = {1, 0, 0, 0};    // orientation offset (wxyz)
    const char* prefix     = "";              // prefix for gripper names (avoids conflicts)
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
bool attach_gripper(const char* arm_mjcf, const GripperSpec* g, const char* out_path);

/*
 * Build a MuJoCo world from one or more URDF robots into a single mjModel/mjData.
 * Also injects any table and objects declared in spec->table and spec->objects.
 * @param[out] out_model  Newly allocated MuJoCo model; caller must free via destroy_scene().
 * @param[out] out_data   Newly allocated MuJoCo data; caller must free via destroy_scene().
 * @param[in]  spec       Scene description: robots, table, objects, timestep, gravity, floor.
 * @return true on success, false on any load or compile error.
 */
bool build_scene(mjModel** out_model, mjData** out_data, const SceneSpec* spec);

/*
 * Free a model/data pair allocated by build_scene().
 * @param[in] model  Model to free (may be null).
 * @param[in] data   Data to free (may be null).
 */
void destroy_scene(mjModel* model, mjData* data);

/*
 * Attach a KDL chain to an already-loaded scene (shared model/data — not owned).
 * Resolves joint names using `prefix` to disambiguate robots in multi-robot scenes.
 * @param[out] s          State to populate (chain, joint maps, window fields zeroed).
 * @param[in]  model      MuJoCo model from build_scene(); not freed by cleanup().
 * @param[in]  data       MuJoCo data from build_scene(); not freed by cleanup().
 * @param[in]  urdf_path  Path to the robot URDF (used to build the KDL chain).
 * @param[in]  base_link  Name of the chain's root link in the URDF.
 * @param[in]  tip_link   Name of the chain's end-effector link in the URDF.
 * @param[in]  prefix     Optional MuJoCo joint-name prefix (default ""); must match SceneRobot::prefix.
 * @return true on success, false if the chain or any joint cannot be found.
 */
bool init_robot(State* s, mjModel* model, mjData* data,
                const char* urdf_path, const char* base_link,
                const char* tip_link,  const char* prefix = "");

/*
 * Open a GLFW window and initialise MuJoCo rendering contexts in s.
 * Must be called after init_robot() or init().
 * @param[in,out] s      State whose window/scn/cam/opt/con fields are filled.
 * @param[in]     title  Window title string.
 * @param[in]     width  Window width in pixels.
 * @param[in]     height Window height in pixels.
 * @return true on success, false if GLFW or MuJoCo context creation fails.
 */
bool init_window(State* s, const char* title = "MuJoCo",
                 int width = 1280, int height = 720);

/*
 * Single-robot shortcut: runs build_scene + init_robot + (optionally) init_window.
 * @param[out] s    Fully initialised state on success.
 * @param[in]  cfg  Configuration struct; cfg->headless skips window creation.
 * @return true on success, false on any error.
 */
bool init(State* s, const Config* cfg);

/*
 * Release all resources owned by s (model/data if _owns_model, GLFW window, GL context).
 * @param[in,out] s  State to tear down; all pointers set to null afterwards.
 */
void cleanup(State* s);

/*
 * Advance the simulation by one timestep and call mj_forward().
 * @param[in,out] s  Simulation state.
 */
void step(State* s);

/*
 * Advance the simulation by n timesteps.
 * @param[in,out] s  Simulation state.
 * @param[in]     n  Number of steps.
 */
void step_n(State* s, int n);

/*
 * Reset simulation to the model's keyframe 0 (or default pose).
 * @param[in,out] s  Simulation state.
 */
void reset(State* s);

/*
 * Returns true if the GLFW window is open and not scheduled for closing.
 * Always returns true in headless mode.
 * @param[in] s  Simulation state.
 */
bool is_running(const State* s);

/*
 * Render the current simulation frame to the window (no-op in headless mode).
 * @param[in,out] s  Simulation state.
 * @return true if the window is still open after rendering.
 */
bool render(State* s);

/*
 * Copy MuJoCo qpos into q, reordered to match KDL chain joint order.
 * Resizes q to s->n_joints if needed.
 * @param[in]  s  Simulation state with a valid data pointer.
 * @param[out] q  Joint positions in KDL chain order.
 * @return true on success, false if s->data is null.
 */
bool sync_to_kdl(const State* s, KDL::JntArray& q);

/*
 * Write KDL joint positions into MuJoCo qpos (KDL chain order → MuJoCo addresses).
 * @param[in,out] s  Simulation state.
 * @param[in]     q  Joint positions in KDL chain order; size must equal s->n_joints.
 */
void sync_from_kdl(State* s, const KDL::JntArray& q);

/*
 * Apply joint torques by writing into s->data->qfrc_applied (KDL chain order).
 * @param[in,out] s    Simulation state.
 * @param[in]     tau  Torques in KDL chain order; size must equal s->n_joints.
 */
void set_torques(State* s, const KDL::JntArray& tau);

/*
 * Add an object to the scene by appending it to spec->objects and rebuilding
 * the model. The old model/data are freed; new ones replace them.
 * Any State objects sharing the old model/data become stale — call init_robot()
 * again on the new model/data after this call.
 * @param[in,out] model  Current model pointer; updated to new model on success.
 * @param[in,out] data   Current data pointer; updated to new data on success.
 * @param[in,out] spec   Scene spec; obj is appended to spec->objects.
 * @param[in]     obj    Object to add.
 * @return true on success; model/data and spec->objects unchanged on failure.
 */
bool scene_add_object(mjModel** model, mjData** data,
                      SceneSpec* spec, const SceneObject& obj);

/*
 * Remove a named object from the scene by erasing it from spec->objects and
 * rebuilding the model. The old model/data are freed; new ones replace them.
 * Any State objects sharing the old model/data become stale — call init_robot()
 * again on the new model/data after this call.
 * @param[in,out] model  Current model pointer; updated to new model on success.
 * @param[in,out] data   Current data pointer; updated to new data on success.
 * @param[in,out] spec   Scene spec; named object removed from spec->objects.
 * @param[in]     name   Name of the object to remove.
 * @return true on success; false if name not found or rebuild fails.
 */
bool scene_remove_object(mjModel** model, mjData** data,
                         SceneSpec* spec, const std::string& name);

} // namespace mj_kdl
