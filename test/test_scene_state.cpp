/* test_scene_state.cpp
 * Env scene slots: free-body poses sampled from qpos rather than the derived body frames, scalar
 * scene joints, wrench and actuator slots, what update() reads and applies, and what reset()
 * restores. Self-skips when Menagerie is absent. */

#include <gtest/gtest.h>

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "example_paths.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static constexpr int kFillerBodies = 50;

class SceneStateTest : public testing::Test
{
  protected:
    mj_kdl::SceneSpec spec_;
    mj_kdl::Env       env_;
    mjModel          *model_ = nullptr;
    mjData           *data_  = nullptr;
    mj_kdl::Robot     robot_;

    void SetUp() override
    {
        const std::string mjcf = mj_kdl_examples::find_menagerie_model("kinova_gen3/gen3.xml");
        const std::string cube = mj_kdl_examples::find_asset("cube.xml");
        if (!fs::exists(mjcf) || !fs::exists(cube)) {
            GTEST_SKIP() << "menagerie or bundled assets not found";
            return;
        }

        spec_.timestep   = 0.002;
        spec_.add_floor  = true;
        spec_.add_skybox = true;
        spec_.robots.push_back(mj_kdl::RobotSpec{ .path = mjcf.c_str(), .attachments = {} });
        spec_.objects.push_back(mj_kdl::SceneObject{
          .name = "cube", .mjcf_path = cube, .pos = { 0.6, 0.0, 1.0 }, .fixed = false });
        spec_.objects.push_back(mj_kdl::SceneObject{ .name     = "block",
                                                     .shape    = mj_kdl::Shape::BOX,
                                                     .size     = { 0.02, 0.02, 0.02 },
                                                     .pos      = { -0.6, 0.0, 0.5 },
                                                     .rgba     = { 0.5f, 0.5f, 0.5f, 1.0f },
                                                     .fixed    = true,
                                                     .friction = { 1.0, 0.005, 0.0001 } });
        for (int i = 0; i < kFillerBodies; ++i) {
            spec_.objects.push_back(mj_kdl::SceneObject{ .name     = "filler_" + std::to_string(i),
                                                         .shape    = mj_kdl::Shape::BOX,
                                                         .size     = { 0.01, 0.01, 0.01 },
                                                         .pos      = { -1.0, 0.05 * i, 0.5 },
                                                         .rgba     = { 0.5f, 0.5f, 0.5f, 1.0f },
                                                         .fixed    = true,
                                                         .friction = { 1.0, 0.005, 0.0001 } });
        }

        ASSERT_TRUE(mj_kdl::init_env(&env_, &spec_));
        model_ = env_.model;
        data_  = env_.data;
        ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&robot_, &env_, "base_link", "bracelet_link"));
    }

    int actuator(int joint, int group) const
    {
        const int jid = mj_name2id(model_, mjOBJ_JOINT, robot_.joint_names[joint].c_str());
        for (int a = 0; a < model_->nu; ++a) {
            if (model_->actuator_trnid[2 * a] == jid && model_->actuator_group[a] == group)
                return a;
        }
        return -1;
    }
};

TEST_F(SceneStateTest, FreeBodyPoseAndDerivedFrameAgreeAfterAStep)
{
    mj_kdl::SceneFreeBodySlot *cube = mj_kdl::bind_scene_free_body(&env_.scene, "cube");
    ASSERT_NE(cube, nullptr);

    for (int i = 0; i < 3; ++i) mj_kdl::step(&env_);
    mj_kdl::update(&env_);

    const double *q = data_->qpos + cube->qpos_adr;
    EXPECT_EQ(cube->pose.p.x(), q[0]);
    EXPECT_EQ(cube->pose.p.y(), q[1]);
    EXPECT_EQ(cube->pose.p.z(), q[2]);
    EXPECT_EQ(cube->seq, 1u);

    // The cube is falling, so a frame one step behind qpos would differ by ~1e-4.
    KDL::Frame derived;
    ASSERT_TRUE(mj_kdl::get_body_frame(&env_, "cube", &derived));
    EXPECT_NEAR(derived.p.x(), cube->pose.p.x(), 1e-12);
    EXPECT_NEAR(derived.p.y(), cube->pose.p.y(), 1e-12);
    EXPECT_NEAR(derived.p.z(), cube->pose.p.z(), 1e-12);
}

TEST_F(SceneStateTest, AFrameFollowsAQposWrittenDirectly)
{
    if (!model_) return;
    const int  adr = mj_kdl::bind_scene_free_body(&env_.scene, "cube")->qpos_adr;
    KDL::Frame cube;
    ASSERT_TRUE(mj_kdl::get_body_frame(&env_, "cube", &cube));

    data_->qpos[adr + 2] = 2.5;
    ASSERT_TRUE(mj_kdl::get_body_frame(&env_, "cube", &cube));
    EXPECT_NEAR(cube.p.z(), 2.5, 1e-12);
}

TEST_F(SceneStateTest, StepMatchesMjStepBitwise)
{
    if (!model_) return;
    mjData *reference = mj_makeData(model_);
    mj_copyData(reference, model_, data_);
    for (int i = 0; i < 200; ++i) {
        mj_kdl::step(&env_);
        mj_step(model_, reference);
    }
    EXPECT_EQ(std::memcmp(data_->qpos, reference->qpos, sizeof(mjtNum) * model_->nq), 0);
    EXPECT_EQ(std::memcmp(data_->qvel, reference->qvel, sizeof(mjtNum) * model_->nv), 0);
    mj_deleteData(reference);
}

TEST_F(SceneStateTest, StepHonoursAQposWrittenBetweenSteps)
{
    if (!model_) return;
    const int adr = mj_kdl::bind_scene_free_body(&env_.scene, "cube")->qpos_adr;
    mj_kdl::step(&env_);
    mjData *reference = mj_makeData(model_);
    mj_copyData(reference, model_, data_);

    data_->qpos[adr + 2] = reference->qpos[adr + 2] = 2.0;
    mj_kdl::step(&env_);
    mj_step(model_, reference);
    EXPECT_EQ(data_->qpos[adr + 2], reference->qpos[adr + 2]);

    KDL::Frame cube;
    ASSERT_TRUE(mj_kdl::get_body_frame(&env_, "cube", &cube));
    EXPECT_NEAR(cube.p.z(), data_->qpos[adr + 2], 1e-12);
    mj_deleteData(reference);
}

// Opens a Simulate window, so it is opt-in: --gtest_also_run_disabled_tests.
TEST_F(SceneStateTest, DISABLED_ViewerKeepsUserWrenchesWhileAnotherThreadReads)
{
    if (!model_) return;
    ASSERT_TRUE(mj_kdl::open_viewer(&env_, "viewer lock test"));

    mj_kdl::SceneWrenchSlot *push = mj_kdl::bind_scene_wrench(&env_.scene, "cube");
    ASSERT_NE(push, nullptr);
    push->wrench = KDL::Wrench(KDL::Vector(0.0, 0.0, 5.0), KDL::Vector::Zero());
    const int fz = 6 * push->body_id + 2;

    std::atomic<bool> stop{ false };
    std::thread       reader([&] {
        KDL::Frame frame;
        while (!stop) mj_kdl::get_body_frame(&env_, "cube", &frame);
    });

    int lost = 0;
    for (int i = 0; i < 1000; ++i) {
        mj_kdl::update(&env_);
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // let the render thread run
        if (data_->xfrc_applied[fz] != 5.0) ++lost;
        ASSERT_TRUE(mj_kdl::step(&env_));
    }
    stop = true;
    reader.join();
    EXPECT_EQ(lost, 0) << "the render thread cleared a user wrench";
}

TEST_F(SceneStateTest, BindFreeBodyRejectsFixedUnknownAndDuplicate)
{
    if (!model_) return;
    const auto level = mj_kdl::get_log_level();
    mj_kdl::set_log_level(mj_kdl::LogLevel::NONE);

    EXPECT_EQ(mj_kdl::bind_scene_free_body(&env_.scene, "block"), nullptr);
    EXPECT_EQ(mj_kdl::bind_scene_free_body(&env_.scene, "no_such_body"), nullptr);
    ASSERT_NE(mj_kdl::bind_scene_free_body(&env_.scene, "cube"), nullptr);
    EXPECT_EQ(mj_kdl::bind_scene_free_body(&env_.scene, "cube"), nullptr);

    mj_kdl::set_log_level(level);
}

TEST_F(SceneStateTest, SceneJointTracksQposAndRejectsAFreeJoint)
{
    mj_kdl::SceneJointSlot *slot = mj_kdl::bind_scene_joint(&env_.scene, "joint_4");
    ASSERT_NE(slot, nullptr);

    KDL::JntArray q(robot_.n_joints);
    for (int i = 0; i < robot_.n_joints; ++i) q(i) = 0.0;
    q(3) = -1.25;
    mj_kdl::set_joint_pos(&robot_, q);
    mj_kdl::update(&env_);
    EXPECT_NEAR(slot->position, -1.25, 1e-12);
    EXPECT_EQ(slot->velocity, data_->qvel[slot->dof_adr]);

    const auto level = mj_kdl::get_log_level();
    mj_kdl::set_log_level(mj_kdl::LogLevel::NONE);
    EXPECT_EQ(mj_kdl::bind_scene_joint(&env_.scene, "cube_free"), nullptr);
    mj_kdl::set_log_level(level);
}

TEST_F(SceneStateTest, ASlotAddressSurvivesFurtherBinds)
{
    mj_kdl::SceneWrenchSlot *first = mj_kdl::bind_scene_wrench(&env_.scene, "cube");
    ASSERT_NE(first, nullptr);

    for (int i = 0; i < kFillerBodies; ++i) {
        ASSERT_NE(
          mj_kdl::bind_scene_wrench(&env_.scene, ("filler_" + std::to_string(i)).c_str()), nullptr
        );
    }

    EXPECT_EQ(first, &env_.scene.wrenches.front());
    EXPECT_EQ(first->name, "cube");
    EXPECT_EQ(env_.scene.wrenches.size(), static_cast<std::size_t>(kFillerBodies) + 1);
}

TEST_F(SceneStateTest, UpdateReadsThenAppliesRobotsAndSlots)
{
    if (!model_) return;
    mj_kdl::SceneWrenchSlot *push = mj_kdl::bind_scene_wrench(&env_.scene, "cube");
    ASSERT_NE(push, nullptr);
    push->wrench = KDL::Wrench(KDL::Vector(0.0, 0.0, 2.0), KDL::Vector::Zero());

    ASSERT_TRUE(mj_kdl::set_control_mode(&robot_, mj_kdl::CtrlMode::TORQUE));
    for (int i = 0; i < robot_.n_joints; ++i) robot_.jnt_trq_cmd[i] = 0.1 * (i + 1);
    mj_kdl::step(&env_);
    mj_kdl::update(&env_);

    for (int i = 0; i < robot_.n_joints; ++i) {
        const int jid = mj_name2id(model_, mjOBJ_JOINT, robot_.joint_names[i].c_str());
        EXPECT_EQ(robot_.jnt_pos_msr[i], data_->qpos[model_->jnt_qposadr[jid]]);
        const int motor = actuator(i, 2); // robot 0's TORQUE group
        ASSERT_GE(motor, 0);
        EXPECT_DOUBLE_EQ(data_->ctrl[motor], 0.1 * (i + 1));
        EXPECT_EQ(data_->qfrc_applied[model_->jnt_dofadr[jid]], 0.0);
    }
    EXPECT_EQ(data_->xfrc_applied[6 * push->body_id + 2], 2.0);
}

TEST_F(SceneStateTest, ApplyClearsAWrenchThatIsNoLongerPushed)
{
    mj_kdl::SceneWrenchSlot *slot = mj_kdl::bind_scene_wrench(&env_.scene, "cube");
    ASSERT_NE(slot, nullptr);

    slot->wrench = KDL::Wrench(KDL::Vector(1.0, -2.0, 3.0), KDL::Vector(0.4, 0.5, 0.6));
    mj_kdl::update(&env_);
    const double *applied = data_->xfrc_applied + 6 * slot->body_id;
    EXPECT_EQ(applied[0], 1.0);
    EXPECT_EQ(applied[5], 0.6);

    slot->wrench = KDL::Wrench::Zero();
    mj_kdl::update(&env_);
    for (int k = 0; k < 6; ++k) EXPECT_EQ(applied[k], 0.0) << "component " << k;
}

TEST_F(SceneStateTest, ResetRestoresEverySlot)
{
    if (!model_) return;
    mj_kdl::SceneJointSlot    *joint = mj_kdl::bind_scene_joint(&env_.scene, "joint_4");
    mj_kdl::SceneFreeBodySlot *cube  = mj_kdl::bind_scene_free_body(&env_.scene, "cube");
    mj_kdl::SceneWrenchSlot   *push  = mj_kdl::bind_scene_wrench(&env_.scene, "cube");
    mj_kdl::SceneActuatorSlot *drive = mj_kdl::bind_scene_actuator(&env_.scene, "joint_4");
    ASSERT_NE(joint, nullptr);
    ASSERT_NE(cube, nullptr);
    ASSERT_NE(push, nullptr);
    ASSERT_NE(drive, nullptr);

    push->wrench   = KDL::Wrench(KDL::Vector(1.0, 2.0, 3.0), KDL::Vector(4.0, 5.0, 6.0));
    drive->command = 1e9;
    for (int i = 0; i < 20; ++i) {
        mj_kdl::update(&env_);
        mj_kdl::step(&env_);
    }
    ASSERT_TRUE(drive->saturated);
    ASSERT_GT(joint->seq, 1u);

    mj_kdl::reset(&env_);

    EXPECT_EQ(push->wrench, KDL::Wrench::Zero());
    EXPECT_EQ(drive->command, data_->ctrl[drive->ctrl_id]);
    EXPECT_FALSE(drive->saturated);
    EXPECT_EQ(joint->seq, 1u) << "re-read once from the reset state";
    EXPECT_EQ(joint->position, data_->qpos[joint->qpos_adr]);
    EXPECT_EQ(cube->seq, 1u);
    EXPECT_EQ(cube->pose.p.z(), data_->qpos[cube->qpos_adr + 2]);

    mj_kdl::update(&env_);
    for (int k = 0; k < 6; ++k) EXPECT_EQ(data_->xfrc_applied[6 * push->body_id + k], 0.0);
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
