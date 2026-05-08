/* test_init.cpp
 * Loads the Kinova GEN3 MJCF (menagerie gen3.xml), runs 100 simulation steps,
 * and verifies basic model properties are consistent.
 * Self-skips when third_party/menagerie is absent. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <iostream>
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
    mj_kdl::Robot s;

    void SetUp() override
    {
        fs::path root = repo_root();
        if (!fs::exists(root / "third_party/menagerie")) {
            GTEST_SKIP() << "third_party/menagerie/ not found";
            return;
        }

        std::string mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();

        mj_kdl::SceneSpec sc;
        mj_kdl::RobotSpec r;
        r.path = mjcf.c_str();
        sc.robots.push_back(r);

        ASSERT_TRUE(mj_kdl::build_scene(&model_, &data_, &sc)) << "build_scene() returned false";
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
    TEST_INFO("nq=" << s.model->nq << " nv=" << s.model->nv << " kdl_joints=" << s.n_joints);
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
    TEST_INFO("sim_time=" << s.data->time);
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

    // Record displaced state.
    double pos_before = s.data->qpos[s.kdl_to_mj_qpos[0]];

    mj_kdl::reset(&s);

    double pos_after = s.data->qpos[s.kdl_to_mj_qpos[0]];
    // After reset qpos should be back near the model default (not the displaced value).
    EXPECT_NE(pos_after, pos_before);
    TEST_INFO(
      "reset(): qpos[0] before=" << pos_before << " after=" << pos_after
    );
}

TEST_F(InitTest, ResetSyncsCmdPorts)
{
    // After reset, jnt_pos_cmd must equal measured qpos and jnt_trq_cmd must be zero.
    unsigned n = static_cast<unsigned>(s.n_joints);

    // Set stale commands.
    for (unsigned i = 0; i < n; ++i) {
        s.jnt_pos_cmd[i] = 99.0;
        s.jnt_trq_cmd[i] = 42.0;
    }

    mj_kdl::reset(&s);

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
    s.on_reset = [&](mjModel *, mjData *) { ++call_count; };

    mj_kdl::reset(&s);

    EXPECT_EQ(call_count, 1) << "on_reset was not called by reset()";
}

TEST_F(InitTest, ResetWithoutOnResetCallbackIsNoOp)
{
    // reset() must not crash when on_reset is not set.
    s.on_reset = nullptr;
    EXPECT_NO_FATAL_FAILURE(mj_kdl::reset(&s));
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
