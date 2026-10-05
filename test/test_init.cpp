/* test_init.cpp
 * Env and Robot lifecycle on the Kinova GEN3: init, reset (hook, ports, options), cleanup, an
 * adopted model pair, a chain adopted from outside, two Envs side by side, what a failure says,
 * required spec fields, the floor height, offscreen rendering and the recorder's output path.
 * Tests that need the bundled Gen3 self-skip without it. */

#include "mjkdl/mjkdl.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ex = mjkdl_examples;
namespace fs = std::filesystem;

static std::string gen3_path() { return ex::find_asset("kinova_gen3/gen3.xml"); }

static mjkdl::SceneSpec arm_scene(const std::string &mjcf, const std::string &prefix = "")
{
    mjkdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = false;
    mjkdl::RobotSpec r;
    r.path   = mjcf;
    r.prefix = prefix;
    sc.robots.push_back(r);
    return sc;
}

static mjkdl::SceneObject small_cube()
{
    mjkdl::SceneObject cube;
    cube.name  = "cube";
    cube.shape = mjkdl::Shape::BOX;
    cube.mass  = 0.1;
    for (int k = 0; k < 3; ++k) cube.size[k] = 0.02;
    for (int k = 0; k < 4; ++k) cube.rgba[k] = 1.0f;
    for (int k = 0; k < 3; ++k) cube.friction[k] = 0.5;
    cube.pos[0] = 0.5;
    cube.pos[2] = 0.02;
    return cube;
}

class InitTest : public testing::Test
{
  protected:
    mjkdl::SceneSpec sc_;
    mjkdl::Env       env_;
    mjkdl::Robot     s;

    void SetUp() override
    {
        const std::string mjcf = gen3_path();
        if (!fs::exists(mjcf)) GTEST_SKIP() << mjcf << " not found";
        sc_ = arm_scene(mjcf);
        ASSERT_TRUE(mjkdl::init_env(&env_, &sc_));
        ASSERT_TRUE(mjkdl::init_robot_from_mjcf(&s, &env_, "base_link", "bracelet_link"));
    }

    int joint(int i) const { return mj_name2id(env_.model, mjOBJ_JOINT, s.joint_names[i].c_str()); }
    double qpos(int i) const { return env_.data->qpos[env_.model->jnt_qposadr[joint(i)]]; }
    int    dof(int i) const { return env_.model->jnt_dofadr[joint(i)]; }
    int    servo(int i) const
    {
        for (int a = 0; a < env_.model->nu; ++a) {
            if (env_.model->actuator_trnid[2 * a] == joint(i) && env_.model->actuator_group[a] == 1)
                return a;
        }
        return -1;
    }
};

TEST_F(InitTest, BasicDOF)
{
    EXPECT_EQ(s.n_joints, 7) << "expected 7 KDL joints, got " << s.n_joints;
    EXPECT_EQ(env_.robots.size(), 1u) << "init_robot_from_mjcf() registers the robot";
}

TEST_F(InitTest, SimulationAdvance)
{
    mjkdl::set_joint_pos(&s, ex::home_q(7));
    const double t0 = env_.data->time;
    for (int k = 0; k < 100; ++k) mjkdl::step(&env_);
    EXPECT_NEAR(env_.data->time - t0, 100 * sc_.timestep, 1e-9);
}

TEST_F(InitTest, ResetRestoresTheKeyframePose)
{
    ASSERT_GT(env_.model->nkey, 0) << "the bundled Gen3 has a home keyframe";
    KDL::JntArray q_displaced = ex::home_q(7);
    for (unsigned i = 0; i < 7; ++i) q_displaced(i) += 0.3;
    mjkdl::set_joint_pos(&s, q_displaced);
    for (int k = 0; k < 50; ++k) mjkdl::step(&env_);

    mjkdl::reset(&env_);
    for (int i = 0; i < s.n_joints; ++i) {
        const int adr = env_.model->jnt_qposadr[joint(i)];
        EXPECT_DOUBLE_EQ(qpos(i), env_.model->key_qpos[adr]) << "joint " << i;
    }
    EXPECT_EQ(env_.data->time, 0.0);
}

TEST_F(InitTest, ResetOptionsPickTheKeyframeOrTheModelDefault)
{
    ASSERT_GE(env_.model->nkey, 2) << "the bundled Gen3 has home and retract keyframes";
    mjkdl::ResetOptions options;
    options.keyframe       = 1;
    mjkdl::ResetInfo info = mjkdl::reset(&env_, &options);
    EXPECT_TRUE(info.used_keyframe);
    EXPECT_EQ(info.keyframe, 1);
    for (int i = 0; i < s.n_joints; ++i) {
        const int adr = env_.model->jnt_qposadr[joint(i)];
        EXPECT_DOUBLE_EQ(qpos(i), env_.model->key_qpos[env_.model->nq + adr]) << "joint " << i;
        EXPECT_DOUBLE_EQ(s.jnt_pos_msr[i], qpos(i));
    }

    options.use_keyframe = false;
    info                 = mjkdl::reset(&env_, &options);
    EXPECT_FALSE(info.used_keyframe);
    EXPECT_EQ(info.keyframe, -1);
    for (int i = 0; i < s.n_joints; ++i) {
        const int adr = env_.model->jnt_qposadr[joint(i)];
        EXPECT_DOUBLE_EQ(qpos(i), env_.model->qpos0[adr]) << "joint " << i;
    }
}

TEST_F(InitTest, ResetSyncsCmdPorts)
{
    for (int i = 0; i < s.n_joints; ++i) {
        s.jnt_pos_cmd[i] = 99.0;
        s.jnt_trq_cmd[i] = 42.0;
    }

    mjkdl::reset(&env_);

    for (int i = 0; i < s.n_joints; ++i) {
        EXPECT_DOUBLE_EQ(s.jnt_pos_cmd[i], qpos(i));
        EXPECT_DOUBLE_EQ(s.jnt_trq_cmd[i], 0.0);
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

    mjkdl::reset(&env_);

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
    int call_count = 0;
    env_.on_reset  = [&](mjkdl::ResetContext *) { ++call_count; };
    mjkdl::reset(&env_);

    EXPECT_EQ(call_count, 1) << "on_reset was not called by reset()";
}

TEST_F(InitTest, ResetWithoutAHookStillReseedsTheRobot)
{
    ASSERT_FALSE(env_.on_reset);
    s.jnt_pos_cmd[0]             = 99.0;
    const mjkdl::ResetInfo info = mjkdl::reset(&env_);
    EXPECT_TRUE(info.used_keyframe);
    EXPECT_DOUBLE_EQ(s.jnt_pos_cmd[0], qpos(0));
}

TEST_F(InitTest, EnvResetInvokesHookAndSyncsRobot)
{
    int call_count = 0;
    env_.on_reset  = [&](mjkdl::ResetContext *ctx) {
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

    mjkdl::ResetInfo info = mjkdl::reset(&env_);

    EXPECT_EQ(call_count, 1);
    EXPECT_TRUE(info.used_keyframe);
    EXPECT_EQ(info.keyframe, 0);
    for (int i = 0; i < s.n_joints; ++i) {
        EXPECT_DOUBLE_EQ(s.jnt_pos_msr[i], qpos(i));
        EXPECT_DOUBLE_EQ(s.jnt_pos_cmd[i], qpos(i));
        EXPECT_DOUBLE_EQ(s.jnt_trq_cmd[i], 0.0);
        EXPECT_DOUBLE_EQ(env_.data->qfrc_applied[dof(i)], 0.0);
    }
}

TEST_F(InitTest, ResetKeepsARequestedControlMode)
{
    s.ctrl_mode = mjkdl::CtrlMode::TORQUE;
    mjkdl::reset(&env_);
    EXPECT_EQ(s.ctrl_mode, mjkdl::CtrlMode::TORQUE);

    mjkdl::update(&env_);
    EXPECT_TRUE(env_.model->opt.disableactuator & (1 << 1)) << "POSITION group off";
    EXPECT_FALSE(env_.model->opt.disableactuator & (1 << 2)) << "TORQUE group on";
}

TEST_F(InitTest, OnResetPrimesCommandsAndMovesAreReadBack)
{
    env_.on_reset = [&](mjkdl::ResetContext *ctx) {
        s.jnt_trq_cmd[0] = 1.5;
        ctx->data->qpos[ctx->model->jnt_qposadr[joint(1)]] += 0.2;
    };
    mjkdl::reset(&env_);
    EXPECT_DOUBLE_EQ(s.jnt_trq_cmd[0], 1.5) << "primed in the hook, not overwritten";
    EXPECT_DOUBLE_EQ(s.jnt_pos_msr[1], qpos(1)) << "a move in the hook is read back";
}

TEST_F(InitTest, CleanupRobotUnregistersIt)
{
    const int a = servo(0);
    ASSERT_GE(a, 0);
    mjkdl::cleanup(&s);
    EXPECT_TRUE(env_.robots.empty());

    env_.data->ctrl[a] = 0.123;
    s.jnt_pos_cmd.assign(7, 1.0);
    mjkdl::update(&env_);
    EXPECT_EQ(env_.data->ctrl[a], 0.123) << "update() no longer commands the robot";
}

TEST_F(InitTest, ChainFromOutsideDrivesTheSameJoints)
{
    mjkdl::Robot given;
    ASSERT_TRUE(mjkdl::init_robot_from_chain(&given, &env_, s.chain, s.joint_names));
    EXPECT_EQ(env_.robots.size(), 2u);
    EXPECT_EQ(given.n_joints, s.n_joints);
    EXPECT_EQ(given.chain.getNrOfSegments(), s.chain.getNrOfSegments());

    KDL::JntArray q = ex::home_q(7);
    q(1) += 0.1;
    mjkdl::set_joint_pos(&given, q);
    mjkdl::update(&env_);
    for (int i = 0; i < 7; ++i) {
        EXPECT_DOUBLE_EQ(given.jnt_pos_msr[i], q(i));
        EXPECT_DOUBLE_EQ(s.jnt_pos_msr[i], q(i)) << "both robots read the same joints";
    }

    std::vector<std::string> short_list(s.joint_names.begin(), s.joint_names.end() - 1);
    mjkdl::Robot            wrong;
    EXPECT_FALSE(mjkdl::init_robot_from_chain(&wrong, &env_, s.chain, short_list));
    mjkdl::cleanup(&given);
}

TEST(TwoEnvs, StepIndependently)
{
    const std::string mjcf = gen3_path();
    if (!fs::exists(mjcf)) GTEST_SKIP() << mjcf << " not found";

    mjkdl::SceneSpec sc = arm_scene(mjcf);
    sc.add_floor         = false;

    mjkdl::Env   a, b;
    mjkdl::Robot ra, rb;
    ASSERT_TRUE(mjkdl::init_env(&a, &sc));
    ASSERT_TRUE(mjkdl::init_env(&b, &sc));
    ASSERT_TRUE(mjkdl::init_robot_from_mjcf(&ra, &a, "base_link", "bracelet_link"));
    ASSERT_TRUE(mjkdl::init_robot_from_mjcf(&rb, &b, "base_link", "bracelet_link"));

    KDL::JntArray q = ex::home_q(7);
    mjkdl::set_joint_pos(&ra, q);
    q(1) += 0.5;
    mjkdl::set_joint_pos(&rb, q);

    KDL::Frame fa, fb;
    ASSERT_TRUE(mjkdl::get_body_frame(&a, "bracelet_link", &fa));
    ASSERT_TRUE(mjkdl::get_body_frame(&b, "bracelet_link", &fb));
    EXPECT_GT((fa.p - fb.p).Norm(), 0.05) << "each Env computes its own frames";

    for (int k = 0; k < 20; ++k) mjkdl::step(&a);
    EXPECT_GT(a.data->time, 0.0);
    EXPECT_EQ(b.data->time, 0.0) << "stepping one Env leaves the other alone";

    KDL::Frame fb_again;
    ASSERT_TRUE(mjkdl::get_body_frame(&b, "bracelet_link", &fb_again));
    EXPECT_TRUE(KDL::Equal(fb, fb_again, 1e-12));
}

TEST(EnvSpec, OwnsItsStringsAcrossARebuild)
{
    const std::string mjcf = gen3_path();
    if (!fs::exists(mjcf)) GTEST_SKIP() << mjcf << " not found";

    mjkdl::Env   env;
    mjkdl::Robot robot;
    {
        const mjkdl::SceneSpec sc = arm_scene(std::string(mjcf), std::string("arm_"));
        ASSERT_TRUE(mjkdl::init_env(&env, &sc));
    }
    ASSERT_TRUE(mjkdl::init_robot_from_mjcf(&robot, &env, "arm_base_link", "arm_bracelet_link"));

    const mjkdl::Status added = mjkdl::scene_add_object(&env, small_cube());
    ASSERT_TRUE(added) << added.error;
    EXPECT_EQ(env.spec.robots[0].prefix, "arm_");

    for (int k = 0; k < 10; ++k) ASSERT_TRUE(mjkdl::step(&env));
    mjkdl::update(&env);
    EXPECT_GT(env.data->time, 0.0);
    EXPECT_EQ(robot.n_joints, 7);
    mjkdl::cleanup(&env);
}

TEST(EnvAdopt, RunsOnTheCallersPairAndNeverFreesIt)
{
    const std::string mjcf = gen3_path();
    if (!fs::exists(mjcf)) GTEST_SKIP() << mjcf << " not found";

    std::vector<std::pair<mjModel *, mjData *>> owned;
    mjkdl::Env                                 env;
    mjkdl::Robot                               robot;
    env.adopt = [&](mjModel *m, mjData *d) {
        mjModel *om = mj_copyModel(nullptr, m);
        mjData  *od = mj_makeData(om);
        mj_copyData(od, m, d);
        mjkdl::destroy_scene(m, d);
        owned.emplace_back(om, od);
        return owned.back();
    };

    const mjkdl::SceneSpec sc = arm_scene(mjcf);
    ASSERT_TRUE(mjkdl::init_env(&env, &sc));
    ASSERT_EQ(owned.size(), 1u);
    EXPECT_EQ(env.model, owned[0].first);
    ASSERT_TRUE(mjkdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link"));
    for (int k = 0; k < 10; ++k) ASSERT_TRUE(mjkdl::step(&env));

    ASSERT_TRUE(mjkdl::scene_add_object(&env, small_cube()));
    ASSERT_EQ(owned.size(), 2u);
    EXPECT_EQ(env.model, owned[1].first);
    EXPECT_EQ(robot.model, owned[1].first);
    EXPECT_GT(owned[1].first->nbody, owned[0].first->nbody);
    for (int k = 0; k < 10; ++k) ASSERT_TRUE(mjkdl::step(&env));

    const fs::path xml = fs::temp_directory_path() / "mjkdl_adopted.xml";
    EXPECT_TRUE(mjkdl::save_model_xml(env.model, xml.c_str())) << "the spec follows the pair";
    fs::remove(xml);

    mjkdl::cleanup(&env);
    // Both pairs are still the caller's: usable here, freed once here (ASan sees a double free).
    for (auto &[m, d] : owned) {
        mj_step(m, d);
        mj_deleteData(d);
        mj_deleteModel(m);
    }
}

TEST(SceneFloor, PlacedAtFloorZ)
{
    mjkdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.floor_z    = -0.72;
    sc.add_skybox = false;

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    ASSERT_TRUE(mjkdl::build_scene(&model, &data, &sc));

    // By type, not by name: the ground plane is deliberately unnamed so it cannot collide
    // with an asset that has a geom called "floor".
    int floor_id = -1;
    for (int i = 0; i < model->ngeom; ++i) {
        if (model->geom_type[i] == mjGEOM_PLANE) floor_id = i;
    }
    ASSERT_GE(floor_id, 0);
    EXPECT_DOUBLE_EQ(model->geom_pos[3 * floor_id + 2], -0.72);

    mjkdl::destroy_scene(model, data);
}

TEST(Recorder, OutputPathReachesFfmpegVerbatim)
{
    mjkdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = false;
    mjkdl::Env env;
    ASSERT_TRUE(mjkdl::init_env(&env, &sc));

    const fs::path dir = fs::temp_directory_path() / "mjkdl_rec_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path out    = dir / "a \"quoted\" $(touch injected) name.mp4";
    const fs::path marker = dir / "injected";

    mjkdl::VideoRecorder vr;
    if (!mjkdl::init_video_recorder(&vr, env.model, out.c_str(), 64, 48, 10))
        GTEST_SKIP() << "no EGL or ffmpeg";
    for (int i = 0; i < 5; ++i) ASSERT_TRUE(mjkdl::record_frame(&vr, &env));
    mjkdl::cleanup(&vr);

    EXPECT_TRUE(fs::exists(out)) << out;
    EXPECT_GT(fs::file_size(out), 0u);
    EXPECT_FALSE(fs::exists(marker)) << "the path went through a shell";
    EXPECT_FALSE(fs::exists(fs::current_path() / "injected"));
    fs::remove_all(dir);
}

TEST(Offscreen, RendersTheSceneIntoABuffer)
{
    mjkdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = true;
    mjkdl::Env env;
    ASSERT_TRUE(mjkdl::init_env(&env, &sc));

    constexpr int         kW = 64, kH = 48;
    mjkdl::VideoRecorder vr;
    if (!mjkdl::init_offscreen(&vr, env.model, kW, kH)) GTEST_SKIP() << "no EGL";
    std::vector<std::uint8_t> rgb(kW * kH * 3, 0);
    ASSERT_TRUE(mjkdl::render_rgb(&vr, &env, rgb.data()));
    mjkdl::cleanup(&vr);

    std::size_t lit = 0;
    for (std::uint8_t c : rgb) lit += c != 0;
    EXPECT_GT(lit, rgb.size() / 2) << "the floor and sky fill the frame";
}

TEST_F(InitTest, AFailureSaysWhy)
{
    mjkdl::Robot  other;
    mjkdl::Status s = mjkdl::init_robot_from_mjcf(&other, &env_, "no_such_body", "bracelet_link");
    EXPECT_FALSE(s);
    EXPECT_NE(s.error.find("no_such_body"), std::string::npos) << s.error;

    mjkdl::SceneObject cube;
    cube.name  = "unweighed_cube";
    cube.shape = mjkdl::Shape::BOX;
    s          = mjkdl::scene_add_object(&env_, cube);
    EXPECT_FALSE(s);
    EXPECT_NE(s.error.find("unweighed_cube"), std::string::npos) << s.error;
    s = mjkdl::scene_remove_object(&env_, "no_such_object");
    EXPECT_FALSE(s);
    EXPECT_NE(s.error.find("no_such_object"), std::string::npos) << s.error;
}

TEST(SceneSpecRequired, AnUnsetFieldFailsTheBuild)
{
    const auto builds = [](const mjkdl::SceneSpec &sc) {
        mjModel   *model = nullptr;
        mjData    *data  = nullptr;
        const bool ok    = static_cast<bool>(mjkdl::build_scene(&model, &data, &sc));
        mjkdl::destroy_scene(model, data);
        return ok;
    };
    mjkdl::SceneSpec base;
    base.timestep   = 0.002;
    base.add_floor  = false;
    base.add_skybox = false;

    mjkdl::SceneObject cube = small_cube();
    cube.friction[0]         = 1.0;
    cube.friction[1]         = 0.005;
    cube.friction[2]         = 0.0001;

    mjkdl::SceneSpec complete = base;
    complete.objects           = { cube };
    EXPECT_TRUE(builds(complete));

    mjkdl::SceneSpec no_mass = complete;
    no_mass.objects[0].mass   = NAN;
    EXPECT_FALSE(builds(no_mass));

    mjkdl::SceneSpec no_friction      = complete;
    no_friction.objects[0].friction[1] = NAN;
    EXPECT_FALSE(builds(no_friction));

    mjkdl::CameraSpec cam;
    cam.name   = "cam";
    cam.pos[0] = cam.pos[1]   = 0.0;
    cam.pos[2]                = 1.0;
    mjkdl::SceneSpec no_fovy = base;
    no_fovy.cameras.push_back(cam);
    EXPECT_FALSE(builds(no_fovy));

    mjkdl::SceneSpec no_timestep = base;
    no_timestep.timestep          = NAN;
    EXPECT_FALSE(builds(no_timestep));
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
