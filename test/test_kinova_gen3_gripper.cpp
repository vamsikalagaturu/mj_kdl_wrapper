// test_kinova_gen3_gripper.cpp
// Attach Robotiq 2F-85 gripper to Kinova Gen3 using attach_gripper(),
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
//   Uses MuJoCo's built-in simulate UI (File/Option/Physics/Rendering/etc panels).
//   Physics thread runs gravity-compensation + constant 2 Nm on joint_7
//   + continuous gripper open/close every 3 s.
//   Left-drag a selected body to push it; right-drag to torque it.
//
// Usage: test_kinova_gen3_gripper [--gui]

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

// MuJoCo simulate UI (from MuJoCo source tree)
#include "simulate.h"
#include "glfw_adapter.h"
#include "array_safety.h"

#include <tinyxml2.h>

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

namespace fs = std::filesystem;
namespace mj = ::mujoco;

static fs::path repo_root()
{
    return fs::path(__FILE__).parent_path().parent_path();
}

// ---------------------------------------------------------------------------
// Patch the combined XML to add floor, sky texture, a directional light,
// and contact exclusions between the arm's bracelet_link and gripper bodies.
// Called once after attach_gripper() generates the file.
// ---------------------------------------------------------------------------
static bool patch_scene(const std::string& path)
{
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) return false;
    auto* root = doc.FirstChildElement("mujoco");
    if (!root) return false;

    // -- asset: skybox gradient + checker floor texture -----------------------
    auto* asset = root->FirstChildElement("asset");
    if (!asset) {
        asset = doc.NewElement("asset");
        root->InsertFirstChild(asset);
    }
    {
        auto* sky = doc.NewElement("texture");
        sky->SetAttribute("type",    "skybox");
        sky->SetAttribute("builtin", "gradient");
        sky->SetAttribute("rgb1",    "0.3 0.45 0.65");
        sky->SetAttribute("rgb2",    "0.65 0.8 0.95");
        sky->SetAttribute("width",   "200");
        sky->SetAttribute("height",  "200");
        asset->InsertFirstChild(sky);

        auto* mat_tex = doc.NewElement("texture");
        mat_tex->SetAttribute("type",    "2d");
        mat_tex->SetAttribute("name",    "groundplane");
        mat_tex->SetAttribute("builtin", "checker");
        mat_tex->SetAttribute("rgb1",    "0.2 0.3 0.4");
        mat_tex->SetAttribute("rgb2",    "0.1 0.2 0.3");
        mat_tex->SetAttribute("width",   "300");
        mat_tex->SetAttribute("height",  "300");
        asset->InsertEndChild(mat_tex);

        auto* mat = doc.NewElement("material");
        mat->SetAttribute("name",        "groundplane");
        mat->SetAttribute("texture",     "groundplane");
        mat->SetAttribute("texrepeat",   "5 5");
        mat->SetAttribute("reflectance", "0.2");
        asset->InsertEndChild(mat);
    }

    // -- worldbody: light + floor ---------------------------------------------
    auto* wb = root->FirstChildElement("worldbody");
    if (!wb) return false;
    {
        auto* light = doc.NewElement("light");
        light->SetAttribute("pos",         "0 0 4");
        light->SetAttribute("directional", "true");
        wb->InsertFirstChild(light);

        auto* floor = doc.NewElement("geom");
        floor->SetAttribute("name",     "floor");
        floor->SetAttribute("type",     "plane");
        floor->SetAttribute("material", "groundplane");
        floor->SetAttribute("size",     "5 5 0.05");
        floor->SetAttribute("condim",   "3");
        wb->InsertAfterChild(light, floor);
    }

    // -- contact: exclude bracelet_link from colliding with gripper bodies ---
    auto* contact = root->FirstChildElement("contact");
    if (!contact) {
        contact = doc.NewElement("contact");
        root->InsertEndChild(contact);
    }
    {
        const char* gripper_bodies[] = {
            "g_base", "g_left_driver", "g_right_driver",
            "g_left_spring_link", "g_right_spring_link",
            "g_left_follower", "g_right_follower",
            "g_left_coupler", "g_right_coupler",
            "g_left_pad", "g_right_pad"
        };
        for (const char* gb : gripper_bodies) {
            auto* exc = doc.NewElement("exclude");
            exc->SetAttribute("body1", "bracelet_link");
            exc->SetAttribute("body2", gb);
            contact->InsertEndChild(exc);
        }
    }

    return doc.SaveFile(path.c_str()) == tinyxml2::XML_SUCCESS;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    bool gui = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--gui") gui = true;

    const fs::path root        = repo_root();
    const std::string arm_mjcf  = (root / "assets/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf  = (root / "assets/robotiq_2f85_v4/2f85.xml").string();
    const std::string combined  = (root / "assets/gen3_with_2f85.xml").string();

    // Test 1: combine arm + gripper
    // Attachment follows the MuJoCo menagerie gen3_with_2f85 reference:
    //   pos  = (0, 0, -0.061525)  — flange face along bracelet_link -z
    //   quat = (0, 1, 0, 0)       — 180° around x so gripper faces away from arm
    mj_kdl::GripperSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = "bracelet_link";
    gs.prefix    = "g_";
    gs.pos[0] = 0.0; gs.pos[1] = 0.0; gs.pos[2] = -0.061525;
    gs.quat[0] = 0.0; gs.quat[1] = 1.0; gs.quat[2] = 0.0; gs.quat[3] = 0.0;

    if (!mj_kdl::attach_gripper(arm_mjcf.c_str(), &gs, combined.c_str())) {
        std::cerr << "FAIL: attach_gripper\n"; return 1;
    }

    if (!patch_scene(combined)) {
        std::cerr << "FAIL: patch_scene\n"; return 1;
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

    // Test 3: gravity torques vs MuJoCo qfrc_bias at q=0
    {
        KDL::JntArray q0(n);
        mj_kdl::sync_from_kdl(&s, q0);
        mj_forward(model, data);

        KDL::JntArray g(n);
        if (dyn.JntToGravity(q0, g) < 0) {
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

    // Test 4: FK sanity check
    KDL::JntArray q_test(n);
    for (unsigned i = 0; i < n; ++i)
        q_test(i) = (i % 2 == 0 ? 1.0 : -1.0) * 30.0 * M_PI / 180.0;

    mj_kdl::sync_from_kdl(&s, q_test);
    mj_forward(model, data);

    KDL::Frame fk_pose;
    fk.JntToCart(q_test, fk_pose);
    double ee_dist = fk_pose.p.Norm();

    std::cout << std::fixed << std::setprecision(3)
              << "EE distance from base at test pose: " << ee_dist * 1000.0 << " mm\n";

    if (ee_dist < 0.1 || ee_dist > 1.1) {
        std::cerr << "FAIL: EE distance " << ee_dist << " m outside expected [0.1, 1.1]\n";
        mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
    }
    std::cout << "  OK\n";

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

    if (std::abs(jrange_hi - 0.9) > 0.01 || jrange_lo < -0.01) {
        std::cerr << "FAIL: unexpected driver joint range\n";
        mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
    }
    std::cout << "  OK\n\nOK\n";

    // -----------------------------------------------------------------------
    // GUI: MuJoCo simulate UI with physics thread
    // -----------------------------------------------------------------------
    if (gui) {
        mj_resetData(model, data);
        mj_forward(model, data);

        // Per-joint addresses for the 7 arm joints
        std::vector<int> arm_qpos(n), arm_dof(s.kdl_to_mj_dof.begin(), s.kdl_to_mj_dof.end());
        for (unsigned i = 0; i < n; ++i)
            arm_qpos[i] = model->jnt_qposadr[model->dof_jntid[arm_dof[i]]];

        // Prime position actuators to current pose so arm doesn't snap at startup
        for (unsigned i = 0; i < n; ++i)
            data->ctrl[i] = data->qpos[arm_qpos[i]];

        mjvCamera cam; mjv_defaultCamera(&cam);
        mjvOption opt; mjv_defaultOption(&opt);
        mjvPerturb pert; mjv_defaultPerturb(&pert);

        auto sim = std::make_unique<mj::Simulate>(
            std::make_unique<mj::GlfwAdapter>(),
            &cam, &opt, &pert, /*is_passive=*/false
        );

        // Load must happen from physics thread AFTER RenderLoop() starts —
        // Simulate::Load() blocks on cond_loadrequest until the render thread
        // calls LoadOnRenderThread().
        std::thread phys([&]() {
            sim->LoadMessage(combined.c_str());
            sim->Load(model, data, combined.c_str());
            {
                std::unique_lock<std::recursive_mutex> lock(sim->mtx);
                mj_forward(model, data);
            }

            while (!sim->exitrequest.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                std::unique_lock<std::recursive_mutex> lock(sim->mtx);

                if (!sim->run) { mj_forward(model, data); continue; }

                // Null arm position actuators so qfrc_applied drives the joints
                for (unsigned i = 0; i < n; ++i)
                    data->ctrl[i] = data->qpos[arm_qpos[i]];

                // Gravity compensation for arm joints via KDL
                KDL::JntArray q(n), g(n);
                for (unsigned i = 0; i < n; ++i) q(i) = data->qpos[arm_qpos[i]];
                dyn.JntToGravity(q, g);
                g(6) += 2.0;  // constant 2 Nm on joint_7
                for (unsigned i = 0; i < n; ++i)
                    data->qfrc_applied[arm_dof[i]] = g(i);

                // Gripper: open/close every 3 s (0 = open, 200 = close)
                data->ctrl[fingers_act] = (std::fmod(data->time, 6.0) < 3.0) ? 200.0 : 0.0;

                if (sim->pert.active) mjv_applyPerturbForce(model, data, &sim->pert);
                mj_step(model, data);
                sim->AddToHistory();
            }
        });

        sim->RenderLoop();  // blocks until window closes
        phys.join();
    }

    mj_kdl::cleanup(&s);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
