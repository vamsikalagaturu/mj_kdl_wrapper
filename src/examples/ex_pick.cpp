/* ex_pick.cpp
 * Scripted pick-and-place: Kinova GEN3 + Robotiq 2F-85 picks an orange cube
 * from the floor and lifts it.
 *
 * IK solves pre-grasp -> grasp -> lift waypoints from home pose.
 * Each phase is time-parameterised with linear joint interpolation and gravity
 * compensation via qfrc_bias.
 *
 * Requires third_party/menagerie (MuJoCo Menagerie submodule).
 *
 * Usage:
 *   ex_pick [--headless]
 *
 * With --headless runs the full 10.5 s scripted sequence and prints final
 * cube height. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <tinyxml2.h>

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

static constexpr double kHomePose[7]  = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };
static constexpr double kCubeX        = 0.4;
static constexpr double kCubeY        = 0.0;
static constexpr double kCubeHS       = 0.02; // half-size of 4 cm cube
static constexpr double kCubeZ        = kCubeHS;
static constexpr double kGripperReach = 0.122; // finger reach along gripper Z

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path().parent_path(); }

static double clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }

static void lerp_q(const KDL::JntArray &a, const KDL::JntArray &b, double t, KDL::JntArray &out)
{
    for (unsigned i = 0; i < a.rows(); ++i) out(i) = a(i) + t * (b(i) - a(i));
}

static bool patch_contact_exclusions(const std::string &path)
{
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) return false;
    auto *root = doc.FirstChildElement("mujoco");
    if (!root) return false;

    auto *contact = root->FirstChildElement("contact");
    if (!contact) {
        contact = doc.NewElement("contact");
        root->InsertFirstChild(contact);
    }

    struct Pair
    {
        const char *body1;
        const char *body2;
    };
    static const Pair kExclude[] = {
        { "bracelet_link", "g_base" },
        { "bracelet_link", "g_left_pad" },
        { "bracelet_link", "g_right_pad" },
        { "half_arm_2_link", "g_base" },
    };
    for (const auto &p : kExclude) {
        auto *ex = doc.NewElement("exclude");
        ex->SetAttribute("body1", p.body1);
        ex->SetAttribute("body2", p.body2);
        contact->InsertEndChild(ex);
    }
    return doc.SaveFile(path.c_str()) == tinyxml2::XML_SUCCESS;
}

struct Phase
{
    double               t_start;
    double               duration;
    const KDL::JntArray *q_from;
    const KDL::JntArray *q_to;
    double               gripper; // 0 = open, 255 = closed
};

int main(int argc, char *argv[])
{
    bool headless = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--headless") headless = true;

    const fs::path root = repo_root();
    if (!fs::exists(root / "third_party/menagerie")) {
        std::cerr << "third_party/menagerie/ not found  - run: "
                     "git submodule update --init third_party/menagerie\n";
        return 1;
    }

    const std::string arm_mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
    const std::string combined = "/tmp/ex_pick.xml";

    mj_kdl::GripperSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = "bracelet_link";
    gs.prefix    = "g_";
    gs.pos[0]    = 0.0;
    gs.pos[1]    = 0.0;
    gs.pos[2]    = -0.061525;
    gs.quat[0]   = 0.0;
    gs.quat[1]   = 1.0;
    gs.quat[2]   = 0.0;
    gs.quat[3]   = 0.0;

    if (!mj_kdl::attach_gripper(arm_mjcf.c_str(), &gs, combined.c_str())) {
        std::cerr << "attach_gripper() failed\n";
        return 1;
    }
    if (!mj_kdl::patch_mjcf_add_skybox(combined.c_str())
        || !mj_kdl::patch_mjcf_add_floor(combined.c_str())) {
        std::cerr << "patch_mjcf visuals failed\n";
        return 1;
    }
    if (!patch_contact_exclusions(combined)) {
        std::cerr << "patch_contact_exclusions() failed\n";
        return 1;
    }

    mj_kdl::SceneObject cube;
    cube.name    = "cube";
    cube.shape   = mj_kdl::ObjShape::BOX;
    cube.size[0] = cube.size[1] = cube.size[2] = kCubeHS;
    cube.pos[0]                                = kCubeX;
    cube.pos[1]                                = kCubeY;
    cube.pos[2]                                = kCubeZ;
    cube.rgba[0]                               = 1.0f;
    cube.rgba[1]                               = 0.5f;
    cube.rgba[2]                               = 0.0f;
    cube.rgba[3]                               = 1.0f;
    cube.mass                                  = 0.1;
    cube.condim                                = 4;
    cube.friction[0]                           = 0.8;
    cube.friction[1]                           = 0.02;
    cube.friction[2]                           = 0.001;
    if (!mj_kdl::patch_mjcf_add_objects(combined.c_str(), { cube })) {
        std::cerr << "patch_mjcf_add_objects() failed\n";
        return 1;
    }

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    if (!mj_kdl::load_mjcf(&model, &data, combined.c_str())) {
        std::cerr << "load_mjcf() failed\n";
        return 1;
    }

    mj_kdl::Robot robot;
    if (!mj_kdl::init_from_mjcf(&robot, model, data, "base_link", "bracelet_link")) {
        std::cerr << "init_from_mjcf() failed\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    unsigned n           = robot.chain.getNrOfJoints();
    int      fingers_act = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    int      cube_jnt    = mj_name2id(model, mjOBJ_JOINT, "cube_joint");
    int      key_id      = mj_name2id(model, mjOBJ_KEY, "home");

    /* IK setup. */
    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::JntArray                   q_min(n), q_max(n);
    for (unsigned i = 0; i < n; ++i) {
        int jid = model->dof_jntid[robot.kdl_to_mj_dof[i]];
        if (model->jnt_limited[jid]) {
            q_min(i) = model->jnt_range[2 * jid];
            q_max(i) = model->jnt_range[2 * jid + 1];
        } else {
            q_min(i) = -2 * M_PI;
            q_max(i) = 2 * M_PI;
        }
    }
    KDL::ChainIkSolverVel_pinv  ik_vel(robot.chain);
    KDL::ChainIkSolverPos_NR_JL ik(robot.chain, q_min, q_max, fk, ik_vel, 500, 1e-5);

    const double        kGraspZ    = kCubeZ + kGripperReach + 0.02;
    const double        kPreGraspZ = kGraspZ + 0.20;
    const double        kLiftZ     = kGraspZ + 0.30;
    const KDL::Rotation kDownRot   = KDL::Rotation::Identity();

    KDL::JntArray q_home(n), q_pregrasp(n), q_grasp(n), q_lift(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    struct WP
    {
        double               z;
        KDL::JntArray       *out;
        const KDL::JntArray *seed;
    };
    WP wps[] = {
        { kPreGraspZ, &q_pregrasp, &q_home },
        { kGraspZ, &q_grasp, &q_pregrasp },
        { kLiftZ, &q_lift, &q_grasp },
    };
    for (auto &wp : wps) {
        KDL::Frame tgt(kDownRot, KDL::Vector(kCubeX, kCubeY, wp.z));
        if (ik.CartToJnt(*wp.seed, tgt, *wp.out) < 0)
            std::cerr << "IK warning: did not converge for z=" << wp.z << "\n";
    }

    Phase phases[] = {
        { 0.0, 1.0, &q_home, &q_home, 0.0 },
        { 1.0, 2.0, &q_home, &q_pregrasp, 0.0 },
        { 3.0, 2.0, &q_pregrasp, &q_grasp, 0.0 },
        { 5.0, 1.5, &q_grasp, &q_grasp, 255.0 },
        { 6.5, 3.0, &q_grasp, &q_lift, 255.0 },
        { 9.5, 1e9, &q_lift, &q_lift, 255.0 },
    };
    constexpr int kNPhases = static_cast<int>(sizeof(phases) / sizeof(phases[0]));

    /* Reset cube and arm. */
    auto reset_cube = [&](mjData *d) {
        int qadr          = model->jnt_qposadr[cube_jnt];
        d->qpos[qadr + 0] = kCubeX;
        d->qpos[qadr + 1] = kCubeY;
        d->qpos[qadr + 2] = kCubeZ;
        d->qpos[qadr + 3] = 1.0;
        d->qpos[qadr + 4] = d->qpos[qadr + 5] = d->qpos[qadr + 6] = 0.0;
    };

    if (key_id >= 0)
        mj_resetDataKeyframe(model, data, key_id);
    else
        mj_kdl::sync_from_kdl(&robot, q_home);
    reset_cube(data);
    mj_forward(model, data);

    auto control = [&](mjModel * /*m*/, mjData *d) {
        double       t  = d->time;
        const Phase *ph = &phases[kNPhases - 1];
        for (int pi = 0; pi < kNPhases - 1; ++pi)
            if (t < phases[pi + 1].t_start) {
                ph = &phases[pi];
                break;
            }

        double        alpha = clamp01((t - ph->t_start) / ph->duration);
        KDL::JntArray q_target(n);
        lerp_q(*ph->q_from, *ph->q_to, alpha, q_target);

        for (unsigned i = 0; i < n; ++i) {
            int dof              = robot.kdl_to_mj_dof[i];
            d->ctrl[i]           = q_target(i);
            d->qfrc_applied[dof] = d->qfrc_bias[dof];
        }
        d->ctrl[fingers_act] = ph->gripper;
    };

    if (headless) {
        /* Step until end of final phase (9.5 s + epsilon). */
        while (data->time < 9.6) {
            control(model, data);
            mj_kdl::step(&robot);
        }
        int    qadr   = model->jnt_qposadr[cube_jnt];
        double cube_z = data->qpos[qadr + 2];
        std::cout << "cube Z after pick: " << std::fixed << std::setprecision(3) << cube_z
                  << " m\n";
    } else {
        mj_kdl::run_simulate_ui(model, data, combined.c_str(), control);
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
