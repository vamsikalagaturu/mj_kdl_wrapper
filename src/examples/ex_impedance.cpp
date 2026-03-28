/* ex_impedance.cpp
 * Cartesian-impedance-like joint-space controller on Kinova GEN3 + Robotiq 2F-85.
 *
 * Control law (per joint):
 *   tau[i] = Kp[i]*(q_home[i] - q[i]) - Kd[i]*qdot[i] + g_kdl[i]
 * where g_kdl is computed via KDL::ChainDynParam::JntToGravity (includes gripper inertia).
 * Applied via TORQUE mode (qfrc_applied).  The gripper cycles open and closed every 3 s.
 *
 * Requires third_party/menagerie (MuJoCo Menagerie submodule).
 *
 * Usage:
 *   ex_impedance [--headless]
 *
 * With --headless runs 200 steps and prints EE drift. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

// Impedance gains  - tuned for Gen3 joint sizes.
static constexpr double kKp[7] = { 100, 200, 100, 200, 100, 200, 100 };
static constexpr double kKd[7] = { 10, 20, 10, 20, 10, 20, 10 };


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

    mj_kdl::AttachmentSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = "bracelet_link";
    gs.prefix    = "g_";
    gs.pos[2]    = -0.061525;
    gs.euler[0]  = 180.0; // 180 deg around X to flip gripper

    mj_kdl::RobotSpec rs;
    rs.path = arm_mjcf.c_str();
    rs.attachments.push_back(gs);

    mj_kdl::SceneSpec sc;
    sc.robots.push_back(rs);

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    if (!mj_kdl::build_scene(&model, &data, &sc)) {
        std::cerr << "build_scene() failed\n";
        return 1;
    }

    mj_kdl::Robot robot;
    if (!mj_kdl::init_robot_from_mjcf(
          &robot, model, data, "base_link", "bracelet_link", "", "g_base"
        )) {
        std::cerr << "init_robot_from_mjcf() failed\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    unsigned n           = robot.chain.getNrOfJoints();
    int      fingers_act = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    int      key_id      = mj_name2id(model, mjOBJ_KEY, "home");

    KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0.0, 0.0, -9.81));
    KDL::JntArray      q_home(n), q(n), g(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

    auto seed_reset_state = [&](mjData *d) {
        for (unsigned i = 0; i < n; ++i) {
            robot.jnt_pos_cmd[i]                    = d->qpos[robot.kdl_to_mj_qpos[i]];
            robot.jnt_trq_cmd[i]                    = 0.0;
            d->qfrc_applied[robot.kdl_to_mj_dof[i]] = 0.0;
        }
        mj_kdl::update(&robot); // seed *_msr and neutralize arm position actuators
        if (fingers_act >= 0) d->ctrl[fingers_act] = 255.0;
    };

    auto reset_to_home = [&](mjData *d) {
        if (key_id >= 0) {
            mj_resetDataKeyframe(model, d, key_id);
        } else {
            mj_resetData(model, d);
            mj_kdl::set_joint_pos(&robot, q_home, false);
        }
        mj_forward(model, d);
        seed_reset_state(d);
    };

    reset_to_home(data);

    double prev_sim_time = data->time;

    /*
     * Per-step sequence:
     *   1. update()  -- refresh *_msr from current state, write old jnt_trq_cmd to qfrc_applied
     *   2. Reset check -- if sim time went backward, zero torques and return
     *   3. Compute jnt_trq_cmd from fresh *_msr
     *   4. Write jnt_trq_cmd directly to qfrc_applied (overrides stale values from step 1)
     *   5. Set gripper ctrl
     */
    auto step_impedance = [&]() {
        mj_kdl::update(
          &robot
        ); // refresh *_msr; old jnt_trq_cmd -> qfrc_applied (overwritten below)

        if (data->time < prev_sim_time - 1e-6) {
            // Simulation was reset: restore the home state and re-seed commands.
            reset_to_home(data);
            prev_sim_time = data->time;
            return;
        }
        prev_sim_time = data->time;

        for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
        dyn.JntToGravity(q, g);
        for (unsigned i = 0; i < n; ++i) {
            robot.jnt_trq_cmd[i] =
              kKp[i] * (kHomePose[i] - robot.jnt_pos_msr[i]) - kKd[i] * robot.jnt_vel_msr[i] + g(i);
        }
        data->ctrl[fingers_act] = (std::fmod(data->time, 6.0) < 3.0) ? 255.0 : 0.0;
    };

    if (headless) {
        KDL::ChainFkSolverPos_recursive fk(robot.chain);
        KDL::JntArray                   q0(n);
        for (unsigned i = 0; i < n; ++i) q0(i) = robot.jnt_pos_cmd[i];
        KDL::Frame ee_start;
        fk.JntToCart(q0, ee_start);

        for (int step = 0; step < 200; ++step) {
            step_impedance();
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
        mj_kdl::Viewer viewer;
        if (!mj_kdl::init_window_sim(&viewer, &robot)) {
            std::cerr << "init_window_sim() failed\n";
            mj_kdl::cleanup(&robot);
            mj_kdl::destroy_scene(model, data);
            return 1;
        }

        while (true) {
            step_impedance();
            if (!mj_kdl::tick(&viewer, model, data)) break;
        }

        mj_kdl::cleanup(&viewer);
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
