/* ex_pos_ctrl.cpp  (MJCF)
 * Joint position control: drive the Kinova GEN3 from the home pose to a target
 * pose using linearly interpolated position setpoints, then hold.
 *
 * The setpoint trajectory is a straight-line interpolation in joint space over
 * kMotionDuration seconds.  After the motion completes the final position is
 * held indefinitely.
 *
 * Requires third_party/menagerie submodule.
 *
 * Usage:
 *   ex_pos_ctrl [--headless]
 *
 * With --headless runs the full motion and prints final max joint error. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

static constexpr double kHomePose[7]    = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };
static constexpr double kTargetPose[7]  = { 0.3, 0.5, 2.9, -2.0, 0.3, 1.2, 1.3 };
static constexpr double kMotionDuration = 2.0; // seconds for the interpolated move

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path().parent_path(); }

int main(int argc, char *argv[])
{
    bool headless = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--headless") headless = true;

    const fs::path root = repo_root();
    if (!fs::exists(root / "third_party/menagerie")) {
        std::cerr << "third_party/menagerie/ not found - run:\n"
                     "  cmake -B build -DFETCH_MENAGERIE=ON\n";
        return 1;
    }

    const std::string mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();

    mj_kdl::SceneSpec sc;
    mj_kdl::RobotSpec r;
    r.path = mjcf.c_str();
    sc.robots.push_back(r);

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    if (!mj_kdl::build_scene(&model, &data, &sc)) {
        std::cerr << "build_scene() failed\n";
        return 1;
    }

    mj_kdl::Robot robot;
    if (!mj_kdl::init_robot_from_mjcf(&robot, model, data, "base_link", "bracelet_link")) {
        std::cerr << "init_robot_from_mjcf() failed\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    unsigned n = static_cast<unsigned>(robot.n_joints);

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    robot.ctrl_mode = mj_kdl::CtrlMode::POSITION;

    double t_start = 0.0;

    auto reset_to_home = [&]() {
        mj_resetData(model, data);
        mj_kdl::set_joint_pos(&robot, q_home, false);
        mj_forward(model, data);
        t_start = data->time;
        for (unsigned i = 0; i < n; ++i) { robot.jnt_pos_cmd[i] = kHomePose[i]; }
        mj_kdl::update(&robot);
    };

    reset_to_home();

    auto ctrl_step = [&]() {
        mj_kdl::update(&robot);

        double alpha = std::clamp((data->time - t_start) / kMotionDuration, 0.0, 1.0);
        for (unsigned i = 0; i < n; ++i)
            robot.jnt_pos_cmd[i] = kHomePose[i] + alpha * (kTargetPose[i] - kHomePose[i]);
    };

    if (headless) {
        // Run until the trajectory is complete plus a short settling period.
        const double end_time = kMotionDuration + 1.0;
        while (data->time < end_time) {
            ctrl_step();
            mj_kdl::step(&robot);
        }

        double max_err = 0.0;
        for (unsigned i = 0; i < n; ++i)
            max_err = std::max(max_err, std::abs(kTargetPose[i] - robot.jnt_pos_msr[i]));
        std::cout << "max joint error at end: " << std::fixed << std::setprecision(4) << max_err
                  << " rad\n";
    } else {
        mj_kdl::Viewer viewer;
        if (!mj_kdl::init_window_sim(&viewer, &robot)) {
            std::cerr << "init_window_sim() failed\n";
            mj_kdl::cleanup(&robot);
            mj_kdl::destroy_scene(model, data);
            return 1;
        }

        double prev_sim_time = data->time;
        while (true) {
            if (data->time < prev_sim_time - 1e-6) reset_to_home();
            prev_sim_time = data->time;
            ctrl_step();
            if (!mj_kdl::step(&robot)) break;
        }

        mj_kdl::cleanup(&viewer);
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
