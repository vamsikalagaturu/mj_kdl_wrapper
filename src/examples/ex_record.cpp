/* ex_record.cpp  (MJCF)
 * Headless H.264 video recording via VideoRecorder (EGL + ffmpeg pipe).
 *
 * Records the Kinova GEN3 arm holding the home pose via gravity compensation
 * for 5 s at 60 fps, loaded from MuJoCo Menagerie MJCF.
 * No display or GLFW window is needed.
 *
 * Requires third_party/menagerie submodule.
 *
 * Requirements:
 *   - BUILD_RECORDER=ON (default) -- EGL support compiled into the library
 *   - ffmpeg available in PATH    -- sudo apt install ffmpeg
 *
 * Usage:
 *   ex_record [output.mp4] [360p|480p|720p|1080p|2k|4k]
 *
 * Output defaults to sim.mp4; resolution defaults to 1080p. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chaindynparam.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };
static constexpr int    kFps         = 60;
static constexpr double kDuration    = 5.0; // seconds

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path().parent_path(); }

static mj_kdl::VideoResolution parse_resolution(const std::string &s)
{
    if (s == "360p") return mj_kdl::VideoResolution::R360p;
    if (s == "480p") return mj_kdl::VideoResolution::R480p;
    if (s == "720p") return mj_kdl::VideoResolution::R720p;
    if (s == "1080p") return mj_kdl::VideoResolution::R1080p;
    if (s == "2k") return mj_kdl::VideoResolution::R2K;
    if (s == "4k") return mj_kdl::VideoResolution::R4K;
    std::cerr << "Unknown resolution '" << s << "'; using 1080p\n";
    return mj_kdl::VideoResolution::R1080p;
}

int main(int argc, char *argv[])
{
    std::string             outmp4 = "sim.mp4";
    mj_kdl::VideoResolution res    = mj_kdl::VideoResolution::R1080p;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a.size() >= 4 && a.substr(a.size() - 4) == ".mp4")
            outmp4 = a;
        else
            res = parse_resolution(a);
    }

    const fs::path root = repo_root();
    if (!fs::exists(root / "third_party/menagerie")) {
        std::cerr << "third_party/menagerie/ not found - run:\n"
                     "  cmake -B build -DFETCH_MENAGERIE=ON\n";
        return 1;
    }

    const std::string mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();

    // Build scene
    mj_kdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = true;
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

    unsigned           n = static_cast<unsigned>(robot.n_joints);
    KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0.0, 0.0, sc.gravity_z));

    // Set home pose
    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];
    mj_kdl::set_joint_pos(&robot, q_home);
    robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

    // Prime gravity torques so first step gets correct compensation.
    KDL::JntArray q(n), g(n);
    dyn.JntToGravity(q_home, g);
    for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = g(i);

    // Init video recorder
    mj_kdl::VideoRecorder vr;
    if (!mj_kdl::init_video_recorder(&vr, model, outmp4.c_str(), res, kFps)) {
        std::cerr << "init_video_recorder() failed -- is EGL available and ffmpeg installed?\n";
        mj_kdl::cleanup(&robot);
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    // Orbit camera slowly around the arm for a nicer recording.
    vr.cam.elevation = -20.0;
    vr.cam.distance  = 1.8;
    vr.cam.lookat[2] = 0.5;

    const int    total_steps  = static_cast<int>(kDuration / model->opt.timestep);
    const double step_per_deg = 360.0 / total_steps; // full orbit over kDuration
    const int steps_per_frame = std::max(1, static_cast<int>(1.0 / (kFps * model->opt.timestep)));

    std::cout << "Recording " << kDuration << " s to " << outmp4 << "  (" << total_steps
              << " steps, " << kFps << " fps)...\n";

    for (int step = 0; step < total_steps; ++step) {
        // Gravity compensation control
        mj_kdl::update(&robot);
        for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
        dyn.JntToGravity(q, g);
        for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = g(i);
        mj_kdl::step(&robot);

        // Record a frame at the target frame rate.
        if (step % steps_per_frame == 0) {
            vr.cam.azimuth = std::fmod(step * step_per_deg, 360.0);
            if (!mj_kdl::record_frame(&vr, model, data)) {
                std::cerr << "record_frame() failed at step " << step << "\n";
                break;
            }
        }
    }

    mj_kdl::cleanup(&vr); // flushes pipe, finalises MP4

    std::cout << "Saved: " << outmp4 << "\n";

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
