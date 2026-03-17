// test_kinova_gen3_pick.cpp
// Kinova Gen3 + Robotiq 2F-85: pick a cube from the floor.
//
// Cube (4 cm, orange) spawned at (0.4, 0, 0.02) in front of arm.
// Target orientation: KDL::Rotation::Identity() on bracelet_link, which
// makes the gripper approach axis point straight down (gripper is attached
// 180° around X below bracelet_link).
//
// Bracelet_link Z target = cube_center_Z + 0.184
//   (0.062 gs_offset + 0.122 finger reach along gripper Z axis)
//
// Tests:
//   1. Model loads; cube body found.
//   2. KDL chain: 7 arm joints.
//   3. IK converges for pre-grasp, grasp, lift (pos error < 2 mm).
//   4. Headless pick simulation (9.5 s): cube lifted > 0.20 m.
//
// GUI (--gui): scripted pick demo with impedance control.
//
// Usage: test_kinova_gen3_pick [--gui]

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

namespace fs = std::filesystem;

static constexpr double kHomePose[7] = {0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708};

// Gravity-compensation-only gains (used for the hold/GUI modes).
// For trajectory tracking, we use the position actuators directly (ctrl=q_target)
// which have clamped forcerange and are inherently stable.
static constexpr double kKd[7] = { 10, 20, 10, 20, 10, 20, 10 };

// Bracelet_link → finger pad offset along gripper Z (measured from 2F-85 geometry):
//   gs_offset(0.0615) + base_mount(0.007) + base(0.0038) +
//   spring_link_z(0.0609) + follower_z(0.0375) + pad_z(0.01352) = 0.18422 m
static constexpr double kGripperReach = 0.18422;

// Cube spawn
static constexpr double kCubeX  = 0.4;
static constexpr double kCubeY  = 0.0;
static constexpr double kCubeZ  = 0.02;   // centre (bottom at z=0)
static constexpr double kCubeHS = 0.02;   // half-size: 4 cm cube

static fs::path repo_root()
{
    return fs::path(__FILE__).parent_path().parent_path();
}

/* MJCF patching helpers */

static bool patch_contact_exclusions(const std::string& path)
{
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) return false;
    auto* root = doc.FirstChildElement("mujoco");
    if (!root) return false;

    auto* contact = root->FirstChildElement("contact");
    if (!contact) { contact = doc.NewElement("contact"); root->InsertEndChild(contact); }

    const char* gripper_bodies[] = {
        "g_base_mount", "g_base",
        "g_left_driver", "g_right_driver",
        "g_left_spring_link", "g_right_spring_link",
        "g_left_follower", "g_right_follower",
        "g_left_coupler", "g_right_coupler",
        "g_left_pad", "g_right_pad",
        "g_left_silicone_pad", "g_right_silicone_pad"
    };
    for (const char* gb : gripper_bodies) {
        auto* exc = doc.NewElement("exclude");
        exc->SetAttribute("body1", "bracelet_link");
        exc->SetAttribute("body2", gb);
        contact->InsertEndChild(exc);
    }
    return doc.SaveFile(path.c_str()) == tinyxml2::XML_SUCCESS;
}

/* Trajectory helpers */

static double clamp01(double t) { return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t); }

static void lerp_q(const KDL::JntArray& from, const KDL::JntArray& to,
                   double alpha, KDL::JntArray& out)
{
    unsigned n = from.rows();
    out.resize(n);
    for (unsigned i = 0; i < n; ++i)
        out(i) = from(i) + alpha * (to(i) - from(i));
}

/* Main */

int main(int argc, char* argv[])
{
    bool gui = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--gui") gui = true;

    const fs::path root       = repo_root();
    if (!fs::exists(root / "third_party/menagerie")) {
        std::cerr << "SKIP: third_party/menagerie/ not found (gitignored). Run locally with the submodule.\n";
        return 0;
    }
    const std::string arm_mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
    const std::string combined = "/tmp/gen3_with_2f85_pick.xml";

    // Build combined model.
    mj_kdl::GripperSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = "bracelet_link";
    gs.prefix    = "g_";
    gs.pos[0] = 0.0; gs.pos[1] = 0.0; gs.pos[2] = -0.061525;
    gs.quat[0] = 0.0; gs.quat[1] = 1.0; gs.quat[2] = 0.0; gs.quat[3] = 0.0;

    if (!mj_kdl::attach_gripper(arm_mjcf.c_str(), &gs, combined.c_str()))
        { std::cerr << "FAIL: attach_gripper\n"; return 1; }
    if (!mj_kdl::patch_mjcf_visuals(combined.c_str()))
        { std::cerr << "FAIL: patch_mjcf_visuals\n"; return 1; }
    if (!patch_contact_exclusions(combined))
        { std::cerr << "FAIL: patch_contact_exclusions\n"; return 1; }

    mj_kdl::SceneObject cube_obj;
    cube_obj.name       = "cube";
    cube_obj.shape      = mj_kdl::ObjShape::BOX;
    cube_obj.size[0]    = kCubeHS; cube_obj.size[1] = kCubeHS; cube_obj.size[2] = kCubeHS;
    cube_obj.pos[0]     = kCubeX;  cube_obj.pos[1]  = kCubeY;  cube_obj.pos[2]  = kCubeZ;
    cube_obj.rgba[0]    = 1.0f; cube_obj.rgba[1] = 0.5f; cube_obj.rgba[2] = 0.0f; cube_obj.rgba[3] = 1.0f;
    cube_obj.mass       = 0.1;
    cube_obj.condim     = 4;
    cube_obj.friction[0] = 0.8; cube_obj.friction[1] = 0.02; cube_obj.friction[2] = 0.001;

    if (!mj_kdl::patch_mjcf_add_objects(combined.c_str(), {cube_obj}))
        { std::cerr << "FAIL: patch_mjcf_add_objects\n"; return 1; }

    // Test 1: model loads; cube body found.
    mjModel* model = nullptr;
    mjData*  data  = nullptr;
    if (!mj_kdl::load_mjcf(&model, &data, combined.c_str()))
        { std::cerr << "FAIL: load_mjcf\n"; return 1; }

    int cube_bid = mj_name2id(model, mjOBJ_BODY, "cube");
    int cube_jnt = mj_name2id(model, mjOBJ_JOINT, "cube_joint");
    if (cube_bid < 0 || cube_jnt < 0) {
        std::cerr << "FAIL: cube body/joint not found\n";
        mj_kdl::destroy_scene(model, data); return 1;
    }
    std::cout << "Model: nq=" << model->nq << " nbody=" << model->nbody
              << " cube_body=" << cube_bid << "  OK\n";

    // Test 2: KDL chain.
    mj_kdl::State s;
    if (!mj_kdl::init_from_mjcf(&s, model, data, "base_link", "bracelet_link")) {
        std::cerr << "FAIL: init_from_mjcf\n"; mj_kdl::destroy_scene(model, data); return 1;
    }
    unsigned n = s.chain.getNrOfJoints();
    if (n != 7u) {
        std::cerr << "FAIL: expected 7 joints, got " << n << "\n";
        mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
    }
    std::cout << "KDL chain: " << n << " joints  OK\n";

    KDL::ChainFkSolverPos_recursive fk(s.chain);

    KDL::JntArray q_min(n), q_max(n);
    for (unsigned i = 0; i < n; ++i) {
        int jid = model->dof_jntid[s.kdl_to_mj_dof[i]];
        if (model->jnt_limited[jid]) {
            q_min(i) = model->jnt_range[2*jid];
            q_max(i) = model->jnt_range[2*jid+1];
        } else {
            q_min(i) = -2 * M_PI;
            q_max(i) =  2 * M_PI;
        }
    }
    KDL::ChainIkSolverVel_pinv  ik_vel(s.chain);
    KDL::ChainIkSolverPos_NR_JL ik(s.chain, q_min, q_max, fk, ik_vel, 500, 1e-5);

    int fingers_act = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    if (fingers_act < 0) {
        std::cerr << "FAIL: g_fingers_actuator not found\n";
        mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
    }

    // Waypoint bracelet_link Z targets (using exact gripper reach geometry).
    // +0.02 m offset: keeps finger geometry clear of the floor when the gripper
    // is open during approach (open fingers extend further down than closed).
    const double kGraspZ    = kCubeZ + kGripperReach + 0.02;
    const double kPreGraspZ = kGraspZ + 0.20;
    const double kLiftZ     = kGraspZ + 0.30;

    // Target rotation: Identity — bracelet_link Z = world Z → gripper points straight down.
    const KDL::Rotation kDownRot = KDL::Rotation::Identity();

    KDL::JntArray q_home_kdl(n);
    for (unsigned i = 0; i < n; ++i) q_home_kdl(i) = kHomePose[i];

    struct WP { const char* name; double z; };
    WP wps[] = { {"pre-grasp", kPreGraspZ}, {"grasp", kGraspZ}, {"lift", kLiftZ} };

    KDL::JntArray q_pregrasp(n), q_grasp(n), q_lift(n);
    KDL::JntArray* ik_out[] = { &q_pregrasp, &q_grasp, &q_lift };

    // Test 3: IK convergence.
    bool ik_ok = true;
    for (int wi = 0; wi < 3; ++wi) {
        KDL::Frame target(kDownRot, KDL::Vector(kCubeX, kCubeY, wps[wi].z));
        KDL::JntArray& seed = (wi == 0) ? q_home_kdl : *ik_out[wi - 1];
        int ret = ik.CartToJnt(seed, target, *ik_out[wi]);

        KDL::Frame ik_frame;
        fk.JntToCart(*ik_out[wi], ik_frame);
        double err = (ik_frame.p - target.p).Norm() * 1000.0;
        std::cout << std::fixed << std::setprecision(2)
                  << "IK " << wps[wi].name << " z=" << wps[wi].z
                  << " (ret=" << ret << ") pos_err=" << err << " mm";
        if (ret < 0 || err > 2.0) {
            std::cout << "  FAIL\n";
            ik_ok = false;
        } else {
            std::cout << "  OK\n";
        }
    }
    if (!ik_ok) { mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1; }

    // Shared control logic (used by headless test and GUI).

    // Place cube at spawn position (freejoint qpos zeroed by keyframe reset).
    auto reset_cube = [&](mjData* d) {
        int qadr = model->jnt_qposadr[cube_jnt];
        d->qpos[qadr + 0] = kCubeX;
        d->qpos[qadr + 1] = kCubeY;
        d->qpos[qadr + 2] = kCubeZ;
        d->qpos[qadr + 3] = 1.0;
        d->qpos[qadr + 4] = d->qpos[qadr + 5] = d->qpos[qadr + 6] = 0.0;
    };

    struct Phase {
        double t_start, duration;
        const KDL::JntArray* q_from;
        const KDL::JntArray* q_to;
        double gripper;
    };
    Phase phases[] = {
        { 0.0, 1.0, &q_home_kdl, &q_home_kdl,  0.0   },  // hold home
        { 1.0, 2.0, &q_home_kdl, &q_pregrasp,   0.0   },  // → pre-grasp
        { 3.0, 2.0, &q_pregrasp, &q_grasp,       0.0   },  // descend
        { 5.0, 1.5, &q_grasp,    &q_grasp,       255.0 },  // close gripper
        { 6.5, 3.0, &q_grasp,    &q_lift,        255.0 },  // lift
        { 9.5, 1e9, &q_lift,     &q_lift,        255.0 },  // hold
    };
    constexpr int kNPhases = sizeof(phases) / sizeof(phases[0]);

    auto apply_control = [&](mjModel* m, mjData* d) {
        double t = d->time;
        const Phase* ph = &phases[kNPhases - 1];
        for (int pi = 0; pi < kNPhases - 1; ++pi)
            if (t < phases[pi + 1].t_start) { ph = &phases[pi]; break; }

        double alpha = clamp01((t - ph->t_start) / ph->duration);
        KDL::JntArray q_target(n);
        lerp_q(*ph->q_from, *ph->q_to, alpha, q_target);

        for (unsigned i = 0; i < n; ++i) {
            int dof = s.kdl_to_mj_dof[i];
            // Position actuators track q_target (stable: force clamped to forcerange).
            // qfrc_applied adds gravity compensation so the actuator P-term
            // only needs to handle tracking error, not fight gravity.
            d->ctrl[i]           = q_target(i);
            d->qfrc_applied[dof] = d->qfrc_bias[dof];
        }
        d->ctrl[fingers_act] = ph->gripper;
    };

    // Test 4: headless pick simulation — verify cube is lifted.
    {
        int key_id = mj_name2id(model, mjOBJ_KEY, "home");
        if (key_id >= 0) mj_resetDataKeyframe(model, data, key_id);
        else             mj_kdl::sync_from_kdl(&s, q_home_kdl);
        reset_cube(data);
        mj_forward(model, data);

        const double kSimEnd = 10.5;
        const int    kSteps  = static_cast<int>(kSimEnd / model->opt.timestep);

        for (int step = 0; step < kSteps; ++step) {
            apply_control(model, data);
            mj_step(model, data);
        }

        // Cube centre Z after lift hold.
        int qadr = model->jnt_qposadr[cube_jnt];
        double cube_final_z = data->qpos[qadr + 2];

        std::cout << std::fixed << std::setprecision(3)
                  << "Cube Z after pick simulation: " << cube_final_z << " m\n";

        const double kLiftThreshold = 0.20;
        if (cube_final_z < kLiftThreshold) {
            std::cerr << "FAIL: cube Z " << cube_final_z
                      << " < " << kLiftThreshold << " m (not lifted)\n";
            mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
        }
        std::cout << "  OK\n\nOK\n";
    }

    // GUI
    if (gui) {
        int key_id = mj_name2id(model, mjOBJ_KEY, "home");
        if (key_id >= 0) mj_resetDataKeyframe(model, data, key_id);
        else             mj_kdl::sync_from_kdl(&s, q_home_kdl);
        reset_cube(data);
        mj_forward(model, data);

        std::cout << "GUI: pick demo — close window to exit\n";
        mj_kdl::run_simulate_ui(model, data, combined.c_str(),
            [&](mjModel* m, mjData* d) { apply_control(m, d); });
    }

    mj_kdl::cleanup(&s);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
