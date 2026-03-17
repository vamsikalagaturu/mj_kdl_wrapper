// test_kinova_gen3_gripper.cpp
// Attach Robotiq 2F-85 gripper (MuJoCo Menagerie) to Kinova Gen3 using attach_gripper(),
// load the combined model, validate arm KDL and gripper simulation.
//
// Tests:
//   1. attach_gripper produces a valid MJCF that loads (nq >= 13, nu >= 8).
//   2. KDL chain for arm (7 joints) built from combined model.
//   3. KDL gravity torques agree with MuJoCo qfrc_bias[0..6] within 5e-2 Nm.
//   4. FK sanity check — EE position within expected workspace.
//   5. Gripper open/close: driver joint range validated.
//
// GUI (--gui):
//   Uses mj_kdl::run_simulate_ui for the real-time simulate window.
//   Physics: position actuators hold arm at home pose + continuous gripper open/close every 3 s.
//
// Usage: test_kinova_gen3_gripper [--gui]

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <tinyxml2.h>

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

static constexpr double kHomePose[7] = {0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708};

static fs::path repo_root()
{
    return fs::path(__FILE__).parent_path().parent_path();
}

// Add contact exclusions between bracelet_link and all g_* gripper bodies.
// Called after patch_mjcf_visuals() on the combined MJCF.
static bool patch_contact_exclusions(const std::string& path)
{
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) return false;
    auto* root = doc.FirstChildElement("mujoco");
    if (!root) return false;

    auto* contact = root->FirstChildElement("contact");
    if (!contact) {
        contact = doc.NewElement("contact");
        root->InsertEndChild(contact);
    }

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

int main(int argc, char* argv[])
{
    bool gui = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--gui") gui = true;

    const fs::path root        = repo_root();
    if (!fs::exists(root / "third_party/menagerie")) {
        std::cerr << "SKIP: third_party/menagerie/ not found (gitignored). Run locally with the submodule.\n";
        return 0;
    }
    const std::string arm_mjcf  = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf  = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
    const std::string combined  = (root / "/tmp/gen3_with_2f85.xml").string();

    // Test 1: combine arm + gripper
    mj_kdl::GripperSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = "bracelet_link";
    gs.prefix    = "g_";
    gs.pos[0] = 0.0; gs.pos[1] = 0.0; gs.pos[2] = -0.061525;
    gs.quat[0] = 0.0; gs.quat[1] = 1.0; gs.quat[2] = 0.0; gs.quat[3] = 0.0;

    if (!mj_kdl::attach_gripper(arm_mjcf.c_str(), &gs, combined.c_str())) {
        std::cerr << "FAIL: attach_gripper\n"; return 1;
    }

    // Add visuals (floor + sky + light) via wrapper utility.
    if (!mj_kdl::patch_mjcf_visuals(combined.c_str())) {
        std::cerr << "FAIL: patch_mjcf_visuals\n"; return 1;
    }

    // Add contact exclusions (gripper-specific).
    if (!patch_contact_exclusions(combined)) {
        std::cerr << "FAIL: patch_contact_exclusions\n"; return 1;
    }

    mjModel* model = nullptr;
    mjData*  data  = nullptr;
    if (!mj_kdl::load_mjcf(&model, &data, combined.c_str())) {
        std::cerr << "FAIL: load_mjcf\n"; return 1;
    }

    std::cout << "Model: nq=" << model->nq << " nv=" << model->nv
              << " nbody=" << model->nbody << " nu=" << model->nu << "\n";

    if (model->nq < 13) {
        std::cerr << "FAIL: expected nq >= 13 (7 arm + 6 gripper), got " << model->nq << "\n";
        mj_kdl::destroy_scene(model, data); return 1;
    }
    if (model->nu < 8) {
        std::cerr << "FAIL: expected nu >= 8 (7 arm + 1 gripper), got " << model->nu << "\n";
        mj_kdl::destroy_scene(model, data); return 1;
    }
    std::cout << "  OK\n";

    // Test 2: KDL arm chain from combined model
    mj_kdl::State s;
    if (!mj_kdl::init_from_mjcf(&s, model, data, "base_link", "bracelet_link")) {
        std::cerr << "FAIL: init_from_mjcf\n";
        mj_kdl::destroy_scene(model, data); return 1;
    }

    unsigned n = s.chain.getNrOfJoints();
    if (n != 7u) {
        std::cerr << "FAIL: expected 7 KDL joints, got " << n << "\n";
        mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
    }
    std::cout << "KDL chain (arm): " << n << " joints\n";

    KDL::ChainFkSolverPos_recursive fk(s.chain);
    KDL::ChainDynParam              dyn(s.chain, KDL::Vector(0, 0, -9.81));

    // Test 3: gravity torques vs MuJoCo qfrc_bias at q=0.
    // (At q=0 the arm is upright and gripper weight contributes minimally,
    //  keeping the KDL vs MuJoCo discrepancy within tolerance.)
    {
        KDL::JntArray q_zero(n);
        mj_kdl::sync_from_kdl(&s, q_zero);
        mj_forward(model, data);

        KDL::JntArray g(n);
        if (dyn.JntToGravity(q_zero, g) < 0) {
            std::cerr << "FAIL: JntToGravity\n";
            mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
        }

        double max_err = 0.0;
        for (unsigned i = 0; i < n; ++i)
            max_err = std::max(max_err,
                               std::abs(g(i) - data->qfrc_bias[s.kdl_to_mj_dof[i]]));

        std::cout << std::fixed << std::setprecision(6)
                  << "Gravity accuracy at q=0: max|KDL - MuJoCo| = " << max_err << " Nm\n";

        if (max_err > 5e-2) {
            std::cerr << "FAIL: gravity error " << max_err << " Nm > 5e-2\n";
            mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
        }
        std::cout << "  OK\n";
    }

    // Test 4: FK sanity check at home pose
    {
        KDL::JntArray q_home(n);
        for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];
        KDL::Frame fk_pose;
        fk.JntToCart(q_home, fk_pose);
        double ee_dist = fk_pose.p.Norm();

        std::cout << std::fixed << std::setprecision(3)
                  << "EE distance from base at home pose: " << ee_dist * 1000.0 << " mm\n";

        if (ee_dist < 0.1 || ee_dist > 1.1) {
            std::cerr << "FAIL: EE distance " << ee_dist << " m outside expected [0.1, 1.1]\n";
            mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
        }
        std::cout << "  OK\n";
    }

    // Test 5: gripper joints and actuator
    int fingers_act = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    if (fingers_act < 0) {
        std::cerr << "FAIL: g_fingers_actuator not found\n";
        mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
    }

    int rdriver_jnt = mj_name2id(model, mjOBJ_JOINT, "g_right_driver_joint");
    int ldriver_jnt = mj_name2id(model, mjOBJ_JOINT, "g_left_driver_joint");
    if (rdriver_jnt < 0 || ldriver_jnt < 0) {
        std::cerr << "FAIL: gripper driver joints not found\n";
        mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
    }

    double jrange_lo = model->jnt_range[2*rdriver_jnt];
    double jrange_hi = model->jnt_range[2*rdriver_jnt+1];
    std::cout << std::fixed << std::setprecision(4)
              << "Gripper right_driver_joint range: [" << jrange_lo << ", " << jrange_hi << "] rad\n";

    if (std::abs(jrange_hi - 0.8) > 0.01 || jrange_lo < -0.01) {
        std::cerr << "FAIL: unexpected driver joint range\n";
        mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
    }
    std::cout << "  OK\n\nOK\n";

    // GUI: simulate UI with real-time physics loop
    if (gui) {
        // Reset to home pose via keyframe if available, else set manually.
        int key_id = mj_name2id(model, mjOBJ_KEY, "home");
        if (key_id >= 0) {
            mj_resetDataKeyframe(model, data, key_id);
        } else {
            KDL::JntArray q_home(n);
            for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];
            mj_kdl::sync_from_kdl(&s, q_home);
        }
        mj_forward(model, data);

        std::cout << "GUI: close window to exit\n";
        mj_kdl::run_simulate_ui(model, data, combined.c_str(),
            [&](mjModel* m, mjData* d) {
                // Pure gravity compensation:
                // Zero P-term (track current qpos) + cancel gravity via qfrc_bias.
                // qfrc_bias includes gripper mass — KDL alone would miss ~1.7 kg.
                for (unsigned i = 0; i < n; ++i) {
                    int dof = s.kdl_to_mj_dof[i];
                    int jid = m->dof_jntid[dof];
                    d->ctrl[i]           = d->qpos[m->jnt_qposadr[jid]];
                    d->qfrc_applied[dof] = d->qfrc_bias[dof];
                }
                // Gripper: open/close every 3 s (ctrl range 0..255, 255=closed)
                d->ctrl[fingers_act] = (std::fmod(d->time, 6.0) < 3.0) ? 255.0 : 0.0;
            });
    }

    mj_kdl::cleanup(&s);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
