/* test_init.cpp
 * Loads the Kinova GEN3 MJCF (menagerie gen3.xml), runs 100 simulation steps,
 * and verifies basic model properties are consistent.
 * Self-skips when Menagerie is absent. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "example_paths.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

namespace fs = std::filesystem;

TEST(ExamplePaths, StaleMenagerieEnvFallsBackToCache)
{
    const auto tmp = fs::temp_directory_path() / "mj_kdl_wrapper_paths_test";
    fs::remove_all(tmp);
    fs::create_directories(tmp / "cache" / "mj_kdl_wrapper" / "menagerie" / "kinova_gen3");

    const auto model = tmp / "cache" / "mj_kdl_wrapper" / "menagerie" / "kinova_gen3" / "gen3.xml";
    std::ofstream(model) << "<mujoco/>";

    const char       *old_xdg         = std::getenv("XDG_CACHE_HOME");
    const char       *old_menagerie   = std::getenv("MJ_KDL_MENAGERIE");
    const std::string saved_xdg       = old_xdg ? old_xdg : "";
    const std::string saved_menagerie = old_menagerie ? old_menagerie : "";

    setenv("XDG_CACHE_HOME", (tmp / "cache").c_str(), 1);
    setenv("MJ_KDL_MENAGERIE", (tmp / "missing").c_str(), 1);

    EXPECT_EQ(mj_kdl_examples::find_menagerie_model("kinova_gen3/gen3.xml"), model.string());

    old_xdg ? setenv("XDG_CACHE_HOME", saved_xdg.c_str(), 1) : unsetenv("XDG_CACHE_HOME");
    old_menagerie ? setenv("MJ_KDL_MENAGERIE", saved_menagerie.c_str(), 1)
                  : unsetenv("MJ_KDL_MENAGERIE");
    fs::remove_all(tmp);
}

class InitTest : public testing::Test
{
  protected:
    mj_kdl::SceneSpec sc_;
    mj_kdl::Env       env_;
    mj_kdl::Robot     s;

    void SetUp() override
    {
        std::string mjcf = mj_kdl_examples::find_menagerie_model("kinova_gen3/gen3.xml");
        if (!fs::exists(mjcf)) {
            GTEST_SKIP() << mjcf << " not found";
            return;
        }

        sc_.timestep   = 0.002;
        sc_.add_floor  = true;
        sc_.add_skybox = true;
        sc_.robots.push_back(mj_kdl::RobotSpec{ .path = mjcf, .attachments = {} });

        ASSERT_TRUE(mj_kdl::init_env(&env_, &sc_)) << "init_env() returned false";
        ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&s, &env_, "base_link", "bracelet_link"))
          << "init_robot_from_mjcf() returned false";
    }

    int joint(int i) const { return mj_name2id(env_.model, mjOBJ_JOINT, s.joint_names[i].c_str()); }
    double qpos(int i) const { return env_.data->qpos[env_.model->jnt_qposadr[joint(i)]]; }
    int    dof(int i) const { return env_.model->jnt_dofadr[joint(i)]; }
};

TEST_F(InitTest, BasicDOF)
{
    EXPECT_EQ(s.n_joints, 7) << "expected 7 KDL joints, got " << s.n_joints;
    EXPECT_EQ(env_.robots.size(), 1u) << "init_robot_from_mjcf() registers the robot";
}

TEST_F(InitTest, SimulationAdvance)
{
    unsigned      n = static_cast<unsigned>(s.n_joints);
    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];
    mj_kdl::set_joint_pos(&s, q_home);

    const double t0 = env_.data->time;
    for (int k = 0; k < 100; ++k) mj_kdl::step(&env_);
    ASSERT_TRUE(env_.data->time > t0) << "simulation time did not advance after 100 steps";
}

/* reset() tests */

TEST_F(InitTest, ResetRestoresDefaultPose)
{
    // Displace the arm and advance, then reset -- qpos must return to default.
    unsigned      n = static_cast<unsigned>(s.n_joints);
    KDL::JntArray q_displaced(n);
    for (unsigned i = 0; i < n; ++i) q_displaced(i) = kHomePose[i] + 0.3;
    mj_kdl::set_joint_pos(&s, q_displaced);
    for (int k = 0; k < 50; ++k) mj_kdl::step(&env_);

    const double pos_before = qpos(0);
    mj_kdl::reset(&env_);
    EXPECT_NE(qpos(0), pos_before);
}

TEST_F(InitTest, ResetSyncsCmdPorts)
{
    // After reset, jnt_pos_cmd must equal measured qpos and jnt_trq_cmd must be zero.
    unsigned n = static_cast<unsigned>(s.n_joints);
    for (unsigned i = 0; i < n; ++i) {
        s.jnt_pos_cmd[i] = 99.0;
        s.jnt_trq_cmd[i] = 42.0;
    }

    mj_kdl::reset(&env_);

    for (unsigned i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(s.jnt_pos_cmd[i], qpos(i))
          << "jnt_pos_cmd[" << i << "] not synced to qpos after reset";
        EXPECT_DOUBLE_EQ(s.jnt_trq_cmd[i], 0.0)
          << "jnt_trq_cmd[" << i << "] not zeroed after reset";
    }
}

TEST_F(InitTest, ResetRestoresEveryPort)
{
    for (int i = 0; i < s.n_joints; ++i) {
        s.jnt_pos_msr[i]   = 7.0;
        s.jnt_vel_msr[i]   = 7.0;
        s.jnt_vel_cmd[i]   = 3.0;
        s.jnt_trq_cmd[i]   = 3.0;
        s.jnt_saturated[i] = 1;
    }

    mj_kdl::reset(&env_);

    for (int i = 0; i < s.n_joints; ++i) {
        EXPECT_DOUBLE_EQ(s.jnt_pos_msr[i], qpos(i));
        EXPECT_DOUBLE_EQ(s.jnt_vel_msr[i], env_.data->qvel[dof(i)]);
        EXPECT_DOUBLE_EQ(s.jnt_pos_cmd[i], qpos(i));
        EXPECT_EQ(s.jnt_vel_cmd[i], 0.0);
        EXPECT_EQ(s.jnt_trq_cmd[i], 0.0);
        EXPECT_EQ(s.jnt_saturated[i], 0);
    }
}

TEST_F(InitTest, ResetInvokesOnResetCallback)
{
    // on_reset must be called by reset() exactly once.
    int call_count = 0;
    env_.on_reset  = [&](mj_kdl::ResetContext *) { ++call_count; };
    mj_kdl::reset(&env_);

    EXPECT_EQ(call_count, 1) << "on_reset was not called by reset()";
}

TEST_F(InitTest, ResetWithoutOnResetCallbackIsNoOp)
{
    // reset() must not crash when on_reset is not set.
    EXPECT_NO_FATAL_FAILURE(mj_kdl::reset(&env_));
}

TEST_F(InitTest, EnvResetInvokesHookAndSyncsRobot)
{
    int call_count = 0;
    env_.on_reset  = [&](mj_kdl::ResetContext *ctx) {
        ++call_count;
        EXPECT_EQ(ctx->env, &env_);
        EXPECT_EQ(ctx->model, env_.model);
        EXPECT_EQ(ctx->data, env_.data);
    };

    for (int i = 0; i < s.n_joints; ++i) {
        s.jnt_pos_cmd[i]                = 99.0;
        s.jnt_trq_cmd[i]                = 42.0;
        env_.data->qfrc_applied[dof(i)] = 12.0;
    }

    mj_kdl::ResetInfo info = mj_kdl::reset(&env_);

    EXPECT_EQ(call_count, 1);
    if (env_.model->nkey > 0) {
        EXPECT_TRUE(info.used_keyframe);
        EXPECT_EQ(info.keyframe, 0);
    }
    for (int i = 0; i < s.n_joints; ++i) {
        EXPECT_DOUBLE_EQ(s.jnt_pos_msr[i], qpos(i));
        EXPECT_DOUBLE_EQ(s.jnt_pos_cmd[i], qpos(i));
        EXPECT_DOUBLE_EQ(s.jnt_trq_cmd[i], 0.0);
        EXPECT_DOUBLE_EQ(env_.data->qfrc_applied[dof(i)], 0.0);
    }
}

TEST_F(InitTest, ResetKeepsARequestedControlMode)
{
    s.ctrl_mode = mj_kdl::CtrlMode::TORQUE;
    mj_kdl::reset(&env_);
    EXPECT_EQ(s.ctrl_mode, mj_kdl::CtrlMode::TORQUE);

    mj_kdl::update(&env_);
    EXPECT_TRUE(env_.model->opt.disableactuator & (1 << 1)) << "POSITION group off";
    EXPECT_FALSE(env_.model->opt.disableactuator & (1 << 2)) << "TORQUE group on";
}

TEST_F(InitTest, OnResetPrimesCommandsAndMovesAreReadBack)
{
    env_.on_reset = [&](mj_kdl::ResetContext *ctx) {
        s.jnt_trq_cmd[0] = 1.5;
        ctx->data->qpos[ctx->model->jnt_qposadr[joint(1)]] += 0.2;
    };
    mj_kdl::reset(&env_);
    EXPECT_DOUBLE_EQ(s.jnt_trq_cmd[0], 1.5) << "primed in the hook, not overwritten";
    EXPECT_DOUBLE_EQ(s.jnt_pos_msr[1], qpos(1)) << "a move in the hook is read back";
}

TEST_F(InitTest, CleanupRobotUnregistersIt)
{
    mj_kdl::cleanup(&s);
    EXPECT_TRUE(env_.robots.empty());
    EXPECT_NO_FATAL_FAILURE(mj_kdl::update(&env_));
}

TEST(TwoEnvs, StepIndependently)
{
    std::string mjcf = mj_kdl_examples::find_menagerie_model("kinova_gen3/gen3.xml");
    if (!fs::exists(mjcf)) GTEST_SKIP() << mjcf << " not found";

    mj_kdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = false;
    sc.add_skybox = false;
    sc.robots.push_back(mj_kdl::RobotSpec{ .path = mjcf });

    mj_kdl::Env   a, b;
    mj_kdl::Robot ra, rb;
    ASSERT_TRUE(mj_kdl::init_env(&a, &sc));
    ASSERT_TRUE(mj_kdl::init_env(&b, &sc));
    ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&ra, &a, "base_link", "bracelet_link"));
    ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&rb, &b, "base_link", "bracelet_link"));

    KDL::JntArray q(7);
    for (int i = 0; i < 7; ++i) q(i) = kHomePose[i];
    mj_kdl::set_joint_pos(&ra, q);
    q(1) += 0.5;
    mj_kdl::set_joint_pos(&rb, q);

    KDL::Frame fa, fb;
    ASSERT_TRUE(mj_kdl::get_body_frame(&a, "bracelet_link", &fa));
    ASSERT_TRUE(mj_kdl::get_body_frame(&b, "bracelet_link", &fb));
    EXPECT_GT((fa.p - fb.p).Norm(), 0.05) << "each Env computes its own frames";

    for (int k = 0; k < 20; ++k) mj_kdl::step(&a);
    EXPECT_GT(a.data->time, 0.0);
    EXPECT_EQ(b.data->time, 0.0) << "stepping one Env leaves the other alone";

    KDL::Frame fb_again;
    ASSERT_TRUE(mj_kdl::get_body_frame(&b, "bracelet_link", &fb_again));
    EXPECT_TRUE(KDL::Equal(fb, fb_again, 1e-12));
}

TEST(EnvSpec, OwnsItsStringsAcrossARebuild)
{
    if (!fs::exists(mj_kdl_examples::find_menagerie_model("kinova_gen3/gen3.xml")))
        GTEST_SKIP() << "kinova_gen3 not found";

    mj_kdl::Env   env;
    mj_kdl::Robot robot;
    {
        const std::string path   = mj_kdl_examples::find_menagerie_model("kinova_gen3/gen3.xml");
        const std::string prefix = "arm_";
        mj_kdl::SceneSpec sc;
        sc.timestep   = 0.002;
        sc.add_floor  = true;
        sc.add_skybox = false;
        sc.robots.push_back(mj_kdl::RobotSpec{ .path = path, .prefix = prefix });
        ASSERT_TRUE(mj_kdl::init_env(&env, &sc));
    }
    ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&robot, &env, "arm_base_link", "arm_bracelet_link"));

    mj_kdl::SceneObject cube;
    cube.name  = "cube";
    cube.shape = mj_kdl::Shape::BOX;
    cube.mass  = 0.1;
    for (int k = 0; k < 3; ++k) cube.size[k] = 0.02;
    for (int k = 0; k < 4; ++k) cube.rgba[k] = 1.0f;
    for (int k = 0; k < 3; ++k) cube.friction[k] = 0.5;
    cube.has_rgba = true;
    cube.pos[0]   = 0.5;
    cube.pos[2]   = 0.02;

    const mj_kdl::Status added = mj_kdl::scene_add_object(&env, cube);
    ASSERT_TRUE(added) << added.error;
    EXPECT_EQ(env.spec.robots[0].prefix, "arm_");

    for (int k = 0; k < 10; ++k) ASSERT_TRUE(mj_kdl::step(&env));
    mj_kdl::update(&env);
    EXPECT_GT(env.data->time, 0.0);
    EXPECT_EQ(robot.n_joints, 7);
    mj_kdl::cleanup(&env);
}

TEST(SceneFloor, PlacedAtFloorZ)
{
    mj_kdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.floor_z    = -0.72;
    sc.add_skybox = false;

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    ASSERT_TRUE(mj_kdl::build_scene(&model, &data, &sc));

    // By type, not by name: the ground plane is deliberately unnamed so it cannot collide
    // with an asset that has a geom called "floor".
    int floor_id = -1;
    for (int i = 0; i < model->ngeom; ++i) {
        if (model->geom_type[i] == mjGEOM_PLANE) floor_id = i;
    }
    ASSERT_GE(floor_id, 0);
    EXPECT_DOUBLE_EQ(model->geom_pos[3 * floor_id + 2], -0.72);

    mj_kdl::destroy_scene(model, data);
}

TEST(Recorder, OutputPathReachesFfmpegVerbatim)
{
    mj_kdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = false;
    mj_kdl::Env env;
    ASSERT_TRUE(mj_kdl::init_env(&env, &sc));

    const fs::path dir = fs::temp_directory_path() / "mj_kdl_rec_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path out    = dir / "a \"quoted\" $(touch injected) name.mp4";
    const fs::path marker = dir / "injected";

    mj_kdl::VideoRecorder vr;
    if (!mj_kdl::init_video_recorder(&vr, env.model, out.c_str(), 64, 48, 10))
        GTEST_SKIP() << "no EGL or ffmpeg";
    for (int i = 0; i < 5; ++i) ASSERT_TRUE(mj_kdl::record_frame(&vr, &env));
    mj_kdl::cleanup(&vr);

    EXPECT_TRUE(fs::exists(out)) << out;
    EXPECT_GT(fs::file_size(out), 0u);
    EXPECT_FALSE(fs::exists(marker)) << "the path went through a shell";
    EXPECT_FALSE(fs::exists(fs::current_path() / "injected"));
    fs::remove_all(dir);
}

TEST(Recorder, FreeCameraLeavesAFixedCamera)
{
    mj_kdl::VideoRecorder vr;
    vr.cam.type       = mjCAMERA_FIXED;
    vr.cam.fixedcamid = 3;
    mj_kdl::set_free_camera(&vr, 2.0, 90.0, -30.0, { 0.1, 0.2, 0.3 });
    EXPECT_EQ(vr.cam.type, mjCAMERA_FREE);
    EXPECT_EQ(vr.cam.fixedcamid, -1);
    EXPECT_DOUBLE_EQ(vr.cam.distance, 2.0);
    EXPECT_DOUBLE_EQ(vr.cam.lookat[2], 0.3);
}

TEST_F(InitTest, AFailureSaysWhy)
{
    mj_kdl::Robot  other;
    mj_kdl::Status s = mj_kdl::init_robot_from_mjcf(&other, &env_, "no_such_body", "bracelet_link");
    EXPECT_FALSE(s);
    EXPECT_NE(s.error.find("no_such_body"), std::string::npos) << s.error;

    mj_kdl::SceneObject cube;
    cube.name  = "unweighed_cube";
    cube.shape = mj_kdl::Shape::BOX;
    s          = mj_kdl::scene_add_object(&env_, cube);
    EXPECT_FALSE(s);
    EXPECT_NE(s.error.find("unweighed_cube"), std::string::npos) << s.error;
    s = mj_kdl::scene_remove_object(&env_, "no_such_object");
    EXPECT_NE(s.error.find("no_such_object"), std::string::npos) << s.error;
}

TEST(SceneSpecRequired, AnUnsetFieldFailsTheBuild)
{
    const auto builds = [](const mj_kdl::SceneSpec &sc) {
        mjModel   *model = nullptr;
        mjData    *data  = nullptr;
        const bool ok    = static_cast<bool>(mj_kdl::build_scene(&model, &data, &sc));
        mj_kdl::destroy_scene(model, data);
        return ok;
    };
    mj_kdl::SceneSpec base;
    base.timestep   = 0.002;
    base.add_floor  = false;
    base.add_skybox = false;

    mj_kdl::SceneObject cube;
    cube.name  = "cube";
    cube.shape = mj_kdl::Shape::BOX;
    for (int k = 0; k < 3; ++k) cube.size[k] = 0.02;
    for (int k = 0; k < 4; ++k) cube.rgba[k] = 1.0f;
    cube.friction[0] = 1.0;
    cube.friction[1] = 0.005;
    cube.friction[2] = 0.0001;
    cube.mass        = 0.1;

    mj_kdl::SceneSpec complete = base;
    complete.objects           = { cube };
    EXPECT_TRUE(builds(complete));

    mj_kdl::SceneSpec no_mass = complete;
    no_mass.objects[0].mass   = NAN;
    EXPECT_FALSE(builds(no_mass));

    mj_kdl::SceneSpec no_friction      = complete;
    no_friction.objects[0].friction[1] = NAN;
    EXPECT_FALSE(builds(no_friction));

    mj_kdl::SceneSpec no_fovy = base;
    no_fovy.cameras.push_back(mj_kdl::CameraSpec{ .name = "cam", .pos = { 0.0, 0.0, 1.0 } });
    EXPECT_FALSE(builds(no_fovy));

    mj_kdl::SceneSpec no_timestep = base;
    no_timestep.timestep          = NAN;
    EXPECT_FALSE(builds(no_timestep));
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
