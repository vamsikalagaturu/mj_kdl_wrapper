/* ex_impedance.cpp
 * Cartesian-impedance-like joint-space controller on Kinova GEN3 + Robotiq 2F-85.
 *
 * Control law (per joint):
 *   tau[i] = Kp[i]*(q_home[i] - q[i]) - Kd[i]*qdot[i] + qfrc_bias[dof_i]
 * Applied via qfrc_applied; POSITION mode keeps ctrl = current qpos (zero effort).
 * The gripper cycles open and closed every 3 s.
 *
 * Requires third_party/menagerie (MuJoCo Menagerie submodule).
 *
 * Usage:
 *   ex_impedance [--headless]
 *
 * With --headless runs 200 steps and prints EE drift. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

/* Impedance gains  - tuned for Gen3 joint sizes. */
static constexpr double kKp[7] = { 100, 200, 100, 200, 100, 200, 100 };
static constexpr double kKd[7] = { 10, 20, 10, 20, 10, 20, 10 };

static const std::vector<std::pair<std::string, std::string>> kGripperExclusions = {
    { "bracelet_link", "g_base" },
    { "bracelet_link", "g_left_pad" },
    { "bracelet_link", "g_right_pad" },
    { "half_arm_2_link", "g_base" },
};

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path().parent_path(); }

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
    const std::string combined = "/tmp/ex_impedance.xml";

    mj_kdl::GripperSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = "bracelet_link";
    gs.prefix    = "g_";
    gs.pos[2]    = -0.061525;
    gs.euler[0]  = 180.0; // 180 deg around X to flip gripper

    if (!mj_kdl::attach_gripper(arm_mjcf.c_str(), &gs, combined.c_str())) {
        std::cerr << "attach_gripper() failed\n";
        return 1;
    }
    if (!mj_kdl::patch_mjcf_add_skybox(combined.c_str())
        || !mj_kdl::patch_mjcf_add_floor(combined.c_str())) {
        std::cerr << "patch_mjcf visuals failed\n";
        return 1;
    }
    if (!mj_kdl::patch_mjcf_contact_exclusions(combined.c_str(), kGripperExclusions)) {
        std::cerr << "patch_mjcf_contact_exclusions() failed\n";
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
    int      key_id      = mj_name2id(model, mjOBJ_KEY, "home");

    if (key_id >= 0) {
        mj_resetDataKeyframe(model, data, key_id);
    } else {
        KDL::JntArray q_home(n);
        for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];
        mj_kdl::set_joint_pos(&robot, q_home);
    }
    mj_forward(model, data);

    /* POSITION mode: ctrl = jnt_pos_cmd each step (neutralizes position actuators).
     * Initialize pos cmd from actual state so first update() writes correct ctrl. */
    robot.ctrl_mode = mj_kdl::CtrlMode::POSITION;
    for (unsigned i = 0; i < n; ++i) robot.jnt_pos_cmd[i] = data->qpos[robot.kdl_to_mj_qpos[i]];

    auto step_impedance = [&](mjModel * /*m*/, mjData *d) {
        mj_kdl::update(&robot); // ctrl = jnt_pos_cmd (neutralize), reads _msr
        for (unsigned i = 0; i < n; ++i) {
            int dof              = robot.kdl_to_mj_dof[i];
            d->qfrc_applied[dof] = kKp[i] * (kHomePose[i] - robot.jnt_pos_msr[i])
                                   - kKd[i] * robot.jnt_vel_msr[i] + robot.jnt_trq_msr[i];
        }
        robot.jnt_pos_cmd    = robot.jnt_pos_msr; // next step: ctrl = current pos
        d->ctrl[fingers_act] = (std::fmod(d->time, 6.0) < 3.0) ? 255.0 : 0.0;
    };

    if (headless) {
        KDL::ChainFkSolverPos_recursive fk(robot.chain);
        KDL::JntArray                   q0(n);
        for (unsigned i = 0; i < n; ++i) q0(i) = robot.jnt_pos_cmd[i];
        KDL::Frame ee_start;
        fk.JntToCart(q0, ee_start);

        for (int step = 0; step < 200; ++step) {
            step_impedance(model, data);
            mj_kdl::step(&robot);
        }

        KDL::JntArray q_end(n);
        for (unsigned i = 0; i < n; ++i) q_end(i) = robot.jnt_pos_msr[i];
        KDL::Frame ee_end;
        fk.JntToCart(q_end, ee_end);
        double drift = (ee_start.p - ee_end.p).Norm();
        std::cout << "EE drift after 200 steps: " << std::fixed << std::setprecision(3)
                  << drift * 1000.0 << " mm\n";
    } else {
        mj_kdl::run_simulate_ui(model, data, combined.c_str(), step_impedance);
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
