// test_kinova_gen3_pick.cpp
// Kinova Gen3 + Robotiq 2F-85: add a cube on the floor and pick it up.
//
// Adds a 5 cm orange cube at (0.4, 0, 0.025) to the combined model XML,
// then scripts a pick motion via time-phased joint interpolation with
// impedance control (Kp/Kd + gravity comp via qfrc_bias).
//
// Motion phases (GUI):
//   0 – 2.0 s  hold home (gripper open)
//   2 – 5.0 s  home → pre-grasp (above cube)
//   5 – 7.0 s  pre-grasp → grasp (descend to cube)
//   7 – 8.5 s  close gripper
//   8.5–11.5 s grasp → lift (gripper closed)
//   11.5 s+    hold lift
//
// Tests:
//   1. Model loads; cube body found.
//   2. KDL chain: 7 arm joints.
//   3. IK converges for pre-grasp, grasp, and lift waypoints (pos error < 5 mm).
//
// Usage: test_kinova_gen3_pick [--gui]

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "mj_kdl_wrapper/simulate_ui.hpp"

#include <tinyxml2.h>

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

static constexpr double kHomePose[7] = {0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708};

static constexpr double kKp[7] = {100, 200, 100, 200, 100, 200, 100};
static constexpr double kKd[7] = { 10,  20,  10,  20,  10,  20,  10};

static fs::path repo_root()
{
    return fs::path(__FILE__).parent_path().parent_path();
}

// ---------------------------------------------------------------------------
// MJCF patching helpers
// ---------------------------------------------------------------------------

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

// Add a free physics-enabled box to worldbody.
static bool patch_add_cube(const std::string& path, const char* name,
                            double x, double y, double z, double hs)
{
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) return false;
    auto* root = doc.FirstChildElement("mujoco");
    if (!root) return false;

    auto* wb = root->FirstChildElement("worldbody");
    if (!wb) return false;

    char pos_buf[64];
    std::snprintf(pos_buf, sizeof(pos_buf), "%.4f %.4f %.4f", x, y, z);
    char sz_buf[64];
    std::snprintf(sz_buf, sizeof(sz_buf), "%.4f %.4f %.4f", hs, hs, hs);

    auto* body = doc.NewElement("body");
    body->SetAttribute("name", name);
    body->SetAttribute("pos", pos_buf);

    auto* fj = doc.NewElement("freejoint");
    char jname[64]; std::snprintf(jname, sizeof(jname), "%s_joint", name);
    fj->SetAttribute("name", jname);
    body->InsertEndChild(fj);

    auto* geom = doc.NewElement("geom");
    geom->SetAttribute("type", "box");
    geom->SetAttribute("size", sz_buf);
    geom->SetAttribute("mass", "0.1");
    geom->SetAttribute("rgba", "1 0.5 0 1");
    geom->SetAttribute("condim", "4");
    body->InsertEndChild(geom);

    wb->InsertEndChild(body);
    return doc.SaveFile(path.c_str()) == tinyxml2::XML_SUCCESS;
}

// ---------------------------------------------------------------------------
// Trajectory helpers
// ---------------------------------------------------------------------------

static double clamp01(double t) { return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t); }

// Linear interpolation between two joint arrays.
static void lerp_q(const KDL::JntArray& from, const KDL::JntArray& to,
                   double alpha, KDL::JntArray& out)
{
    unsigned n = from.rows();
    out.resize(n);
    for (unsigned i = 0; i < n; ++i)
        out(i) = from(i) + alpha * (to(i) - from(i));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    bool gui = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--gui") gui = true;

    const fs::path root       = repo_root();
    const std::string arm_mjcf = (root / "assets/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
    const std::string combined = "/tmp/gen3_with_2f85_pick.xml";

    // Build combined model and patch.
    mj_kdl::GripperSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = "bracelet_link";
    gs.prefix    = "g_";
    gs.pos[0] = 0.0; gs.pos[1] = 0.0; gs.pos[2] = -0.061525;
    gs.quat[0] = 0.0; gs.quat[1] = 1.0; gs.quat[2] = 0.0; gs.quat[3] = 0.0;

    if (!mj_kdl::attach_gripper(arm_mjcf.c_str(), &gs, combined.c_str()))   { std::cerr << "FAIL: attach_gripper\n"; return 1; }
    if (!mj_kdl::patch_mjcf_visuals(combined.c_str()))                       { std::cerr << "FAIL: patch_mjcf_visuals\n"; return 1; }
    if (!patch_contact_exclusions(combined))                                  { std::cerr << "FAIL: patch_contact_exclusions\n"; return 1; }

    // Cube at (0.4, 0, 0.04): 8 cm cube (half-size=0.04), clearly visible on floor.
    constexpr double kCubeX = 0.4, kCubeY = 0.0, kCubeZ = 0.04;
    if (!patch_add_cube(combined, "cube", kCubeX, kCubeY, kCubeZ, 0.04)) {
        std::cerr << "FAIL: patch_add_cube\n"; return 1;
    }

    // Test 1: model loads; cube body found.
    mjModel* model = nullptr;
    mjData*  data  = nullptr;
    if (!mj_kdl::load_mjcf(&model, &data, combined.c_str())) { std::cerr << "FAIL: load_mjcf\n"; return 1; }

    int cube_bid = mj_name2id(model, mjOBJ_BODY, "cube");
    if (cube_bid < 0) {
        std::cerr << "FAIL: cube body not found\n";
        mj_kdl::destroy_scene(model, data); return 1;
    }
    std::cout << "Model: nq=" << model->nq << " nbody=" << model->nbody
              << " cube_body_id=" << cube_bid << "  OK\n";

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

    // Joint limits for IK (unlimited joints use ±2π).
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

    KDL::ChainIkSolverVel_pinv   ik_vel(s.chain);
    KDL::ChainIkSolverPos_NR_JL  ik(s.chain, q_min, q_max, fk, ik_vel, 500, 1e-5);

    int fingers_act = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    if (fingers_act < 0) {
        std::cerr << "FAIL: g_fingers_actuator not found\n";
        mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
    }

    // Compute IK waypoints.
    // Target rotation: bracelet_link Z aligned with world Z (Identity).
    // With the gripper attached 180° around X below bracelet_link, Identity rotation
    // means the gripper approach axis points straight down — correct for top-down grasp.
    //
    // Waypoint Z values are bracelet_link positions (EE of KDL chain):
    //   bracelet_link Z = cube_center_Z + gripper_mount_offset + finger_reach
    //                   ≈ 0.04 + 0.062 + 0.13 ≈ 0.23 m for the grasp height.
    KDL::JntArray q_home_kdl(n);
    for (unsigned i = 0; i < n; ++i) q_home_kdl(i) = kHomePose[i];

    const KDL::Rotation kDownRot = KDL::Rotation::Identity();

    struct Waypoint { const char* name; double x, y, z; };
    Waypoint waypoints[] = {
        { "pre-grasp", kCubeX, kCubeY, 0.42 },  // bracelet_link above cube
        { "grasp",     kCubeX, kCubeY, 0.23 },  // bracelet_link at grasp height
        { "lift",      kCubeX, kCubeY, 0.55 },  // bracelet_link lifted
    };

    KDL::JntArray q_pregrasp(n), q_grasp(n), q_lift(n);
    KDL::JntArray* ik_targets[] = { &q_pregrasp, &q_grasp, &q_lift };

    // Test 3: IK convergence for all waypoints.
    bool ik_ok = true;
    for (int wi = 0; wi < 3; ++wi) {
        KDL::Frame target(kDownRot, KDL::Vector(waypoints[wi].x, waypoints[wi].y, waypoints[wi].z));
        // Use previous waypoint solution as seed for better convergence.
        KDL::JntArray& seed = (wi == 0) ? q_home_kdl : *ik_targets[wi - 1];
        int ret = ik.CartToJnt(seed, target, *ik_targets[wi]);

        KDL::Frame ik_frame;
        fk.JntToCart(*ik_targets[wi], ik_frame);
        double err = (ik_frame.p - target.p).Norm() * 1000.0;

        std::cout << std::fixed << std::setprecision(2)
                  << "IK " << waypoints[wi].name << " (ret=" << ret
                  << ") pos_err=" << err << " mm\n";
        if (ret < 0 || err > 5.0) {
            std::cerr << "FAIL: IK for " << waypoints[wi].name << "\n";
            ik_ok = false;
        } else {
            std::cout << "  OK\n";
        }
    }
    if (!ik_ok) { mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1; }
    std::cout << "\nOK\n";

    // -----------------------------------------------------------------------
    // GUI: scripted pick demo
    // -----------------------------------------------------------------------
    if (gui) {
        // Reset to home pose.
        int key_id = mj_name2id(model, mjOBJ_KEY, "home");
        if (key_id >= 0) mj_resetDataKeyframe(model, data, key_id);
        else             mj_kdl::sync_from_kdl(&s, q_home_kdl);
        mj_forward(model, data);

        // Motion schedule: (start_time, duration, q_from, q_to, gripper_cmd)
        struct Phase {
            double t_start, duration;
            const KDL::JntArray* q_from;
            const KDL::JntArray* q_to;
            double gripper;   // 0=open, 255=closed
        };
        Phase phases[] = {
            { 0.0,  2.0, &q_home_kdl,  &q_home_kdl,  0.0   },  // hold home
            { 2.0,  3.0, &q_home_kdl,  &q_pregrasp,   0.0   },  // move to pre-grasp
            { 5.0,  2.0, &q_pregrasp,  &q_grasp,      0.0   },  // descend
            { 7.0,  1.5, &q_grasp,     &q_grasp,      255.0 },  // close gripper
            { 8.5,  3.0, &q_grasp,     &q_lift,       255.0 },  // lift
            { 11.5, 1e9, &q_lift,      &q_lift,       255.0 },  // hold
        };
        constexpr int kNPhases = sizeof(phases) / sizeof(phases[0]);

        std::cout << "GUI: pick demo — close window to exit\n";
        mj_kdl::run_simulate_ui(model, data, combined.c_str(),
            [&](mjModel* m, mjData* d) {
                // Find current phase.
                double t = d->time;
                const Phase* ph = &phases[kNPhases - 1];
                for (int pi = 0; pi < kNPhases - 1; ++pi) {
                    if (t < phases[pi + 1].t_start) { ph = &phases[pi]; break; }
                }

                // Interpolated joint target.
                double alpha = clamp01((t - ph->t_start) / ph->duration);
                KDL::JntArray q_target(n);
                lerp_q(*ph->q_from, *ph->q_to, alpha, q_target);

                // Impedance control: Kp*(q_target - q) - Kd*qdot + grav_comp
                for (unsigned i = 0; i < n; ++i) {
                    int dof = s.kdl_to_mj_dof[i];
                    int jid = m->dof_jntid[dof];
                    double q    = d->qpos[m->jnt_qposadr[jid]];
                    double qdot = d->qvel[dof];
                    d->ctrl[i]           = q;  // zero P-error on position actuators
                    d->qfrc_applied[dof] = kKp[i] * (q_target(i) - q)
                                         - kKd[i] * qdot
                                         + d->qfrc_bias[dof];
                }
                d->ctrl[fingers_act] = ph->gripper;
            });
    }

    mj_kdl::cleanup(&s);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
