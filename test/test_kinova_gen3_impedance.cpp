// test_kinova_gen3_impedance.cpp
// Kinova Gen3 + Robotiq 2F-85: home-pose impedance control.
//
// Impedance law (per joint):
//   tau[i] = Kp[i]*(q_home[i] - q[i]) - Kd[i]*qdot[i] + qfrc_bias[dof_i]
// Applied via qfrc_applied; position actuators are zeroed (ctrl = current_qpos).
//
// Tests:
//   1. Model loads with nq >= 13, nu >= 8.
//   2. KDL chain: 7 arm joints.
//   3. 200-step impedance hold at home pose — EE drift < 1 mm.
//
// GUI (--gui):
//   Arm holds home pose via impedance; gripper cycles open/close every 3 s.
//
// Usage: test_kinova_gen3_impedance [--gui]

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "mj_kdl_wrapper/simulate_ui.hpp"

#include <tinyxml2.h>

#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

static constexpr double kHomePose[7] = {0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708};

// Impedance gains — tuned for Gen3 joint sizes.
// Large joints (2,4,6): higher stiffness; small joints (1,3,5,7): lower.
static constexpr double kKp[7] = {100, 200, 100, 200, 100, 200, 100};
static constexpr double kKd[7] = { 10,  20,  10,  20,  10,  20,  10};

static fs::path repo_root()
{
    return fs::path(__FILE__).parent_path().parent_path();
}

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

    const fs::path root       = repo_root();
    const std::string arm_mjcf = (root / "assets/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
    const std::string combined = (root / "assets/gen3_with_2f85_impedance.xml").string();

    // Build combined model.
    mj_kdl::GripperSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = "bracelet_link";
    gs.prefix    = "g_";
    gs.pos[0] = 0.0; gs.pos[1] = 0.0; gs.pos[2] = -0.061525;
    gs.quat[0] = 0.0; gs.quat[1] = 1.0; gs.quat[2] = 0.0; gs.quat[3] = 0.0;

    if (!mj_kdl::attach_gripper(arm_mjcf.c_str(), &gs, combined.c_str())) {
        std::cerr << "FAIL: attach_gripper\n"; return 1;
    }
    if (!mj_kdl::patch_mjcf_visuals(combined.c_str())) {
        std::cerr << "FAIL: patch_mjcf_visuals\n"; return 1;
    }
    if (!patch_contact_exclusions(combined)) {
        std::cerr << "FAIL: patch_contact_exclusions\n"; return 1;
    }

    // Test 1: model loads.
    mjModel* model = nullptr;
    mjData*  data  = nullptr;
    if (!mj_kdl::load_mjcf(&model, &data, combined.c_str())) {
        std::cerr << "FAIL: load_mjcf\n"; return 1;
    }
    if (model->nq < 13 || model->nu < 8) {
        std::cerr << "FAIL: nq=" << model->nq << " nu=" << model->nu << "\n";
        mj_kdl::destroy_scene(model, data); return 1;
    }
    std::cout << "Model: nq=" << model->nq << " nu=" << model->nu << "  OK\n";

    // Test 2: KDL arm chain.
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
    std::cout << "KDL chain: " << n << " joints  OK\n";

    KDL::ChainFkSolverPos_recursive fk(s.chain);

    int fingers_act = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    if (fingers_act < 0) {
        std::cerr << "FAIL: g_fingers_actuator not found\n";
        mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
    }

    // Helper: apply one impedance step and advance simulation.
    auto step_impedance = [&](mjModel* m, mjData* d, const double q_target[7]) {
        for (unsigned i = 0; i < n; ++i) {
            int dof = s.kdl_to_mj_dof[i];
            int jid = m->dof_jntid[dof];
            double q    = d->qpos[m->jnt_qposadr[jid]];
            double qdot = d->qvel[dof];
            d->ctrl[i]           = q;  // zero P-error
            d->qfrc_applied[dof] = kKp[i] * (q_target[i] - q)
                                 - kKd[i] * qdot
                                 + d->qfrc_bias[dof];
        }
        mj_step(m, d);
    };

    // Test 3: impedance hold at home — EE drift < 1 mm over 200 steps.
    {
        KDL::JntArray q_home_kdl(n);
        for (unsigned i = 0; i < n; ++i) q_home_kdl(i) = kHomePose[i];
        mj_kdl::sync_from_kdl(&s, q_home_kdl);
        mj_forward(model, data);

        KDL::Frame ee_init;
        fk.JntToCart(q_home_kdl, ee_init);

        for (int step = 0; step < 200; ++step)
            step_impedance(model, data, kHomePose);

        KDL::JntArray q_now(n);
        mj_kdl::sync_to_kdl(&s, q_now);
        KDL::Frame ee_now;
        fk.JntToCart(q_now, ee_now);
        double drift = (ee_now.p - ee_init.p).Norm() * 1000.0;

        std::cout << std::fixed << std::setprecision(3)
                  << "Impedance hold drift (200 steps): " << drift << " mm\n";
        if (drift > 1.0) {
            std::cerr << "FAIL: drift " << drift << " mm > 1 mm\n";
            mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
        }
        std::cout << "  OK\n\nOK\n";
    }

    // -----------------------------------------------------------------------
    // GUI
    // -----------------------------------------------------------------------
    if (gui) {
        // Reset to home pose.
        int key_id = mj_name2id(model, mjOBJ_KEY, "home");
        if (key_id >= 0) {
            mj_resetDataKeyframe(model, data, key_id);
        } else {
            KDL::JntArray q_home_kdl(n);
            for (unsigned i = 0; i < n; ++i) q_home_kdl(i) = kHomePose[i];
            mj_kdl::sync_from_kdl(&s, q_home_kdl);
        }
        mj_forward(model, data);

        std::cout << "GUI: close window to exit\n";
        mj_kdl::run_simulate_ui(model, data, combined.c_str(),
            [&](mjModel* m, mjData* d) {
                for (unsigned i = 0; i < n; ++i) {
                    int dof = s.kdl_to_mj_dof[i];
                    int jid = m->dof_jntid[dof];
                    double q    = d->qpos[m->jnt_qposadr[jid]];
                    double qdot = d->qvel[dof];
                    d->ctrl[i]           = q;
                    d->qfrc_applied[dof] = kKp[i] * (kHomePose[i] - q)
                                         - kKd[i] * qdot
                                         + d->qfrc_bias[dof];
                }
                d->ctrl[fingers_act] = (std::fmod(d->time, 6.0) < 3.0) ? 255.0 : 0.0;
            });
    }

    mj_kdl::cleanup(&s);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
