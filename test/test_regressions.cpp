// Regressions from the 0.4.0 audit; headless, self-skips without the bundled Gen3.

#include "mj_kdl_wrapper/image_io.hpp"
#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "example_paths.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static mj_kdl::SceneObject make_box(const std::string &name, double x)
{
    mj_kdl::SceneObject box;
    box.name  = name;
    box.shape = mj_kdl::Shape::BOX;
    box.fixed = true;
    for (int k = 0; k < 3; ++k) box.size[k] = 0.02;
    box.pos[0] = x;
    box.pos[1] = 0.5;
    box.pos[2] = 0.02;
    for (int k = 0; k < 4; ++k) box.rgba[k] = k == 0 || k == 3 ? 1.0f : 0.0f;
    box.friction[0] = 1.0;
    box.friction[1] = 0.005;
    box.friction[2] = 0.0001;
    return box;
}

class RebuildTest : public testing::Test
{
  protected:
    mj_kdl::SceneSpec spec_;
    mj_kdl::Env       env_;
    mj_kdl::Robot     robot_;

    void SetUp() override
    {
        const std::string gen3 = mj_kdl_examples::find_menagerie_model("kinova_gen3/gen3.xml");
        if (gen3.empty()) GTEST_SKIP() << "kinova_gen3/gen3.xml not found";
        spec_.timestep   = 0.002;
        spec_.add_floor  = true;
        spec_.add_skybox = false;
        spec_.robots.emplace_back();
        spec_.robots.back().path = gen3;
        ASSERT_TRUE(mj_kdl::init_env(&env_, &spec_));
        ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&robot_, &env_, "base_link", "bracelet_link"));
    }

    double qpos(const char *joint) const
    {
        return env_.data->qpos[env_.model->jnt_qposadr[mj_name2id(env_.model, mjOBJ_JOINT, joint)]];
    }
    double ctrl(const char *actuator) const
    {
        return env_.data->ctrl[mj_name2id(env_.model, mjOBJ_ACTUATOR, actuator)];
    }
};

TEST_F(RebuildTest, AddObjectKeepsThePhysicsStateAndTheCommands)
{
    ASSERT_TRUE(mj_kdl::set_control_mode(&robot_, mj_kdl::CtrlMode::TORQUE));
    for (int k = 0; k < 50; ++k) {
        mj_kdl::update(&env_);
        robot_.jnt_trq_cmd[1] = 5.0;
        mj_kdl::step(&env_);
    }
    mj_kdl::update(&env_);
    const double time   = env_.data->time;
    const double q2     = qpos("joint_2");
    const double u2     = ctrl("joint_2_torque");
    const auto   q_msr  = robot_.jnt_pos_msr;
    const auto   trq_in = robot_.jnt_trq_cmd;
    ASSERT_NE(q2, 0.0);
    ASSERT_NE(u2, 0.0);

    ASSERT_TRUE(mj_kdl::scene_add_object(&env_, make_box("box", 0.5)));
    EXPECT_EQ(env_.data->time, time);
    EXPECT_EQ(qpos("joint_2"), q2);
    EXPECT_EQ(ctrl("joint_2_torque"), u2) << "the command survives the mode switch";
    EXPECT_EQ(robot_.ctrl_mode, mj_kdl::CtrlMode::TORQUE);
    EXPECT_EQ(robot_.jnt_trq_cmd, trq_in);
    mj_kdl::update(&env_);
    EXPECT_EQ(robot_.jnt_pos_msr, q_msr);
}

TEST_F(RebuildTest, FailedRemoveKeepsTheObjectOrder)
{
    spec_.objects.push_back(make_box("base", 0.5));
    auto child      = make_box("child", 0.0);
    child.attach_to = { mj_kdl::AttachKind::Body, "base" };
    spec_.objects.push_back(child);
    ASSERT_TRUE(mj_kdl::init_env(&env_, &spec_));

    EXPECT_FALSE(mj_kdl::scene_remove_object(&env_, "base")) << "child needs its parent";
    ASSERT_EQ(env_.spec.objects.size(), 2u);
    EXPECT_EQ(env_.spec.objects[0].name, "base");
    EXPECT_EQ(env_.spec.objects[1].name, "child");
    EXPECT_TRUE(mj_kdl::scene_add_object(&env_, make_box("more", -0.5)));
}

TEST_F(RebuildTest, RemovingARobotsJointFailsAndLeavesTheEnv)
{
    const fs::path xml = fs::temp_directory_path() / "mj_kdl_regressions_door.xml";
    std::ofstream(xml) << R"(<mujoco><worldbody><body name="door" pos="0 1 0.5">
      <joint name="door_hinge" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.2 0.02 0.3" mass="1"/></body></worldbody></mujoco>)";
    mj_kdl::SceneObject door_obj;
    door_obj.name      = "door";
    door_obj.mjcf_path = xml.string();
    door_obj.fixed     = true;
    ASSERT_TRUE(mj_kdl::scene_add_object(&env_, door_obj));
    KDL::Chain chain;
    chain.addSegment(KDL::Segment("door", KDL::Joint("door_hinge", KDL::Joint::RotZ)));
    mj_kdl::Robot door;
    ASSERT_TRUE(mj_kdl::init_robot_from_chain(&door, &env_, chain, { "door_hinge" }));

    mjModel *const model = env_.model;
    EXPECT_FALSE(mj_kdl::scene_remove_object(&env_, "door"));
    fs::remove(xml);
    EXPECT_EQ(env_.model, model);
    EXPECT_EQ(env_.spec.objects.size(), 1u);
    mj_kdl::update(&env_);
    mj_kdl::step(&env_);
    EXPECT_EQ(door.jnt_pos_msr.size(), 1u);
}

TEST_F(RebuildTest, FailedReinitLeavesTheRobotAsItWas)
{
    mj_kdl::ToolFrameSpec tool;
    tool.tcp_site = "no_such_site";
    EXPECT_FALSE(
      mj_kdl::init_robot_from_mjcf(&robot_, &env_, "base_link", "spherical_wrist_2_link", "", &tool)
    );
    EXPECT_EQ(robot_.n_joints, 7);
    EXPECT_EQ(robot_.chain.getNrOfJoints(), 7u);
    EXPECT_EQ(robot_.jnt_pos_msr.size(), 7u);
    EXPECT_EQ(env_.robots.size(), 1u);
    mj_kdl::update(&env_);
    mj_kdl::step(&env_);
}

TEST_F(RebuildTest, AnOnResetSetBeforeInitEnvIsKept)
{
    mj_kdl::Env env;
    int         calls = 0;
    env.on_reset      = [&calls](mj_kdl::ResetContext *) { ++calls; };
    ASSERT_TRUE(mj_kdl::init_env(&env, &spec_));
    mj_kdl::reset(&env);
    EXPECT_EQ(calls, 1);
}

TEST_F(RebuildTest, RecordersFollowARebuildAndOutliveEachOther)
{
    mj_kdl::VideoRecorder a, b;
    if (!mj_kdl::init_offscreen(&a, env_.model, 64, 48)) GTEST_SKIP() << "no EGL";
    ASSERT_TRUE(mj_kdl::init_offscreen(&b, env_.model, 64, 48));
    std::vector<std::uint8_t> rgb(64 * 48 * 3);
    ASSERT_TRUE(mj_kdl::render_rgb(&a, &env_, rgb.data()));
    ASSERT_TRUE(mj_kdl::render_rgb(&b, &env_, rgb.data()));

    ASSERT_TRUE(mj_kdl::scene_add_object(&env_, make_box("box", 0.5)));
    ASSERT_TRUE(mj_kdl::render_rgb(&a, &env_, rgb.data()));
    const std::vector<std::uint8_t> before = rgb;
    mj_kdl::cleanup(&b);
    std::fill(rgb.begin(), rgb.end(), 0xAB);
    EXPECT_TRUE(mj_kdl::render_rgb(&a, &env_, rgb.data()));
    EXPECT_EQ(rgb, before) << "b's cleanup ended a's context";
    ASSERT_TRUE(mj_kdl::init_offscreen(&a, env_.model, 32, 24)) << "an open recorder re-inits";
    EXPECT_TRUE(mj_kdl::render_rgb(&a, &env_, rgb.data()));
    mj_kdl::cleanup(&a);
}

TEST(LogLevel, IsASeverityThreshold)
{
    const auto level = mj_kdl::get_log_level();
    const auto logs  = [](mj_kdl::LogLevel threshold) {
        mj_kdl::set_log_level(threshold);
        testing::internal::CaptureStderr();
        MJ_LOG_INFO("info");
        MJ_LOG_WARN("warn");
        MJ_LOG_ERROR("error");
        return testing::internal::GetCapturedStderr();
    };
    const std::string info = logs(mj_kdl::LogLevel::INFO);
    EXPECT_NE(info.find("info"), std::string::npos);
    EXPECT_NE(info.find("error"), std::string::npos);
    const std::string warn = logs(mj_kdl::LogLevel::WARN);
    EXPECT_EQ(warn.find("info"), std::string::npos);
    EXPECT_NE(warn.find("warn"), std::string::npos);
    EXPECT_NE(warn.find("error"), std::string::npos);
    const std::string error = logs(mj_kdl::LogLevel::ERROR);
    EXPECT_EQ(error.find("warn"), std::string::npos);
    EXPECT_NE(error.find("error"), std::string::npos);
    EXPECT_TRUE(logs(mj_kdl::LogLevel::NONE).empty());
    mj_kdl::set_log_level(level);
}

TEST(Screenshot, PathReachesFfmpegVerbatimAndAMissingFfmpegIsAFailure)
{
    const fs::path dir = fs::temp_directory_path() / "mj_kdl_png_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path                  out    = dir / "a \"quoted\" $(touch injected) shot.png";
    const fs::path                  marker = dir / "injected";
    const std::vector<std::uint8_t> rgb(8 * 4 * 3, 128);

    if (!mj_kdl::write_png_rgb(out.string(), rgb.data(), 8, 4)) GTEST_SKIP() << "no ffmpeg";
    EXPECT_GT(fs::file_size(out), 0u);
    EXPECT_FALSE(fs::exists(marker));
    EXPECT_FALSE(fs::exists(fs::current_path() / "injected"));

    const std::string path = std::getenv("PATH") ? std::getenv("PATH") : "";
    setenv("PATH", dir.c_str(), 1);
    const std::vector<std::uint8_t> big(1024 * 1024 * 3, 0);
    EXPECT_FALSE(mj_kdl::write_png_rgb((dir / "b.png").string(), big.data(), 1024, 1024));
    setenv("PATH", path.c_str(), 1);
    fs::remove_all(dir);
}

// The display is unreachable, so no window can open; the viewer must say so, not end the process.
TEST_F(RebuildTest, AViewerThatCannotOpenReturnsAnError)
{
    const char       *display       = std::getenv("DISPLAY");
    const char       *wayland       = std::getenv("WAYLAND_DISPLAY");
    const std::string saved_display = display ? display : "";
    const std::string saved_wayland = wayland ? wayland : "";
    setenv("DISPLAY", ":987", 1);
    unsetenv("WAYLAND_DISPLAY");

    const mj_kdl::Status s = mj_kdl::open_viewer(&env_);

    display ? setenv("DISPLAY", saved_display.c_str(), 1) : unsetenv("DISPLAY");
    if (wayland) setenv("WAYLAND_DISPLAY", saved_wayland.c_str(), 1);
    EXPECT_FALSE(s);
    EXPECT_FALSE(mj_kdl::is_running(&env_.viewer));
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
