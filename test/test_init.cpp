/* test_init.cpp
 * Loads the Kinova GEN3 MJCF (menagerie gen3.xml), runs 100 simulation steps,
 * and verifies basic model properties are consistent.
 * Self-skips when third_party/menagerie is absent. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <gtest/gtest.h>

#include <string>
#include <filesystem>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

class InitTest : public testing::Test
{
  protected:
    mjModel      *model_ = nullptr;
    mjData       *data_  = nullptr;
    mj_kdl::SceneSpec sc_;
    mj_kdl::Robot s;

    void SetUp() override
    {
        fs::path root = repo_root();
        std::string mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
        if (!fs::exists(mjcf)) {
            GTEST_SKIP() << mjcf << " not found";
            return;
        }

        sc_.timestep   = 0.002;
        sc_.add_floor  = true;
        sc_.add_skybox = true;
        sc_.robots.push_back(mj_kdl::RobotSpec{ .path = mjcf.c_str(), .attachments = {} });

        ASSERT_TRUE(mj_kdl::build_scene(&model_, &data_, &sc_)) << "build_scene() returned false";
        ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&s, model_, data_, "base_link", "bracelet_link"))
          << "init_robot_from_mjcf() returned false";
    }

    void TearDown() override
    {
        mj_kdl::cleanup(&s);
        mj_kdl::destroy_scene(model_, data_);
    }
};

TEST_F(InitTest, BasicDOF)
{
    EXPECT_EQ(s.n_joints, 7) << "expected 7 KDL joints, got " << s.n_joints;
}

TEST_F(InitTest, SimulationAdvance)
{
    unsigned      n = static_cast<unsigned>(s.n_joints);
    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];
    mj_kdl::set_joint_pos(&s, q_home);
    mj_forward(s.model, s.data);

    const double t0 = s.data->time;
    mj_kdl::step_n(&s, 100);
    ASSERT_TRUE(s.data->time > t0) << "simulation time did not advance after 100 steps";
}

/* reset() tests */

TEST_F(InitTest, ResetRestoresDefaultPose)
{
    // Displace the arm and advance, then reset -- qpos must return to default.
    unsigned      n = static_cast<unsigned>(s.n_joints);
    KDL::JntArray q_displaced(n);
    for (unsigned i = 0; i < n; ++i) q_displaced(i) = kHomePose[i] + 0.3;
    mj_kdl::set_joint_pos(&s, q_displaced);
    mj_kdl::step_n(&s, 50);

    double pos_before = s.data->qpos[s.kdl_to_mj_qpos[0]];

    mj_kdl::Env env;
    env.model = model_;
    env.data  = data_;
    mj_kdl::env_add_robot(&env, &s);
    mj_kdl::reset(&env);

    double pos_after = s.data->qpos[s.kdl_to_mj_qpos[0]];
    EXPECT_NE(pos_after, pos_before);
}

TEST_F(InitTest, ResetSyncsCmdPorts)
{
    // After reset, jnt_pos_cmd must equal measured qpos and jnt_trq_cmd must be zero.
    unsigned n = static_cast<unsigned>(s.n_joints);
    for (unsigned i = 0; i < n; ++i) {
        s.jnt_pos_cmd[i] = 99.0;
        s.jnt_trq_cmd[i] = 42.0;
    }

    mj_kdl::Env env;
    env.model = model_;
    env.data  = data_;
    mj_kdl::env_add_robot(&env, &s);
    mj_kdl::reset(&env);

    for (unsigned i = 0; i < n; ++i) {
        double qpos = s.data->qpos[s.kdl_to_mj_qpos[i]];
        EXPECT_DOUBLE_EQ(s.jnt_pos_cmd[i], qpos)
          << "jnt_pos_cmd[" << i << "] not synced to qpos after reset";
        EXPECT_DOUBLE_EQ(s.jnt_trq_cmd[i], 0.0)
          << "jnt_trq_cmd[" << i << "] not zeroed after reset";
    }
}

TEST_F(InitTest, ResetInvokesOnResetCallback)
{
    // on_reset must be called by reset() exactly once.
    int call_count = 0;

    mj_kdl::Env env;
    env.model    = model_;
    env.data     = data_;
    env.on_reset = [&](mj_kdl::ResetContext *) { ++call_count; };
    mj_kdl::env_add_robot(&env, &s);
    mj_kdl::reset(&env);

    EXPECT_EQ(call_count, 1) << "on_reset was not called by reset()";
}

TEST_F(InitTest, ResetWithoutOnResetCallbackIsNoOp)
{
    // reset() must not crash when on_reset is not set.
    mj_kdl::Env env;
    env.model = model_;
    env.data  = data_;
    mj_kdl::env_add_robot(&env, &s);
    EXPECT_NO_FATAL_FAILURE(mj_kdl::reset(&env));
}

TEST_F(InitTest, EnvResetInvokesHookAndSyncsRobot)
{
    mj_kdl::Env env;
    env.spec  = sc_;
    env.model = model_;
    env.data  = data_;
    mj_kdl::env_add_robot(&env, &s);

    int call_count = 0;
    env.on_reset = [&](mj_kdl::ResetContext *ctx) {
        ++call_count;
        EXPECT_EQ(ctx->env, &env);
        EXPECT_EQ(ctx->model, model_);
        EXPECT_EQ(ctx->data, data_);
    };

    for (int i = 0; i < s.n_joints; ++i) {
        s.jnt_pos_cmd[i] = 99.0;
        s.jnt_trq_cmd[i] = 42.0;
        s.data->qfrc_applied[s.kdl_to_mj_dof[i]] = 12.0;
    }

    mj_kdl::ResetInfo info = mj_kdl::reset(&env);

    EXPECT_EQ(call_count, 1);
    if (model_->nkey > 0) {
        EXPECT_TRUE(info.used_keyframe);
        EXPECT_EQ(info.keyframe, 0);
    }
    for (int i = 0; i < s.n_joints; ++i) {
        double qpos = s.data->qpos[s.kdl_to_mj_qpos[i]];
        EXPECT_DOUBLE_EQ(s.jnt_pos_msr[i], qpos);
        EXPECT_DOUBLE_EQ(s.jnt_pos_cmd[i], qpos);
        EXPECT_DOUBLE_EQ(s.jnt_trq_cmd[i], 0.0);
        EXPECT_DOUBLE_EQ(s.data->qfrc_applied[s.kdl_to_mj_dof[i]], 0.0);
    }

    env.model = nullptr;
    env.data  = nullptr;
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
