/* test_scene_state.cpp
 * SceneState: free-body poses sampled from qpos rather than the derived body frames, scalar
 * scene joints, wrench slots, and the read/apply split of update().
 * Self-skips when Menagerie is absent. */

#include <gtest/gtest.h>

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "example_paths.hpp"

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static constexpr int kFillerBodies = 50;

class SceneStateTest : public testing::Test
{
  protected:
    mjModel           *model_ = nullptr;
    mjData            *data_  = nullptr;
    mj_kdl::SceneSpec  spec_;
    mj_kdl::Robot      robot_;
    mj_kdl::SceneState scene_;

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
        spec_.objects.push_back(mj_kdl::SceneObject{ .name  = "block",
                                                     .shape = mj_kdl::Shape::BOX,
                                                     .size  = { 0.02, 0.02, 0.02 },
                                                     .pos   = { -0.6, 0.0, 0.5 },
                                                     .fixed = true });
        for (int i = 0; i < kFillerBodies; ++i) {
            spec_.objects.push_back(mj_kdl::SceneObject{ .name  = "filler_" + std::to_string(i),
                                                         .shape = mj_kdl::Shape::BOX,
                                                         .size  = { 0.01, 0.01, 0.01 },
                                                         .pos   = { -1.0, 0.05 * i, 0.5 },
                                                         .fixed = true });
        }

        ASSERT_TRUE(mj_kdl::build_scene(&model_, &data_, &spec_));
        ASSERT_TRUE(
          mj_kdl::init_robot_from_mjcf(&robot_, model_, data_, "base_link", "bracelet_link")
        );
        ASSERT_TRUE(mj_kdl::init_scene_state(&scene_, model_));
    }

    void TearDown() override
    {
        if (!model_) return;
        mj_kdl::cleanup(&robot_);
        mj_kdl::destroy_scene(model_, data_);
    }
};

TEST_F(SceneStateTest, FreeBodyPoseAndDerivedFrameAgreeAfterAStep)
{
    mj_kdl::SceneFreeBodySlot *cube = mj_kdl::bind_scene_free_body(&scene_, "cube");
    ASSERT_NE(cube, nullptr);

    for (int i = 0; i < 3; ++i) mj_kdl::step(&robot_);
    mj_kdl::read_scene_state(&scene_, data_);

    const double *q = data_->qpos + cube->qpos_adr;
    EXPECT_EQ(cube->pose.p.x(), q[0]);
    EXPECT_EQ(cube->pose.p.y(), q[1]);
    EXPECT_EQ(cube->pose.p.z(), q[2]);
    EXPECT_EQ(cube->seq, 1u);

    // The cube is falling, so a frame one step behind qpos would differ by ~1e-4.
    KDL::Frame derived;
    ASSERT_TRUE(mj_kdl::get_body_frame(model_, data_, "cube", &derived));
    EXPECT_NEAR(derived.p.x(), cube->pose.p.x(), 1e-12);
    EXPECT_NEAR(derived.p.y(), cube->pose.p.y(), 1e-12);
    EXPECT_NEAR(derived.p.z(), cube->pose.p.z(), 1e-12);
}

TEST_F(SceneStateTest, StepMatchesMjStepBitwise)
{
    if (!model_) return;
    mjData *reference = mj_makeData(model_);
    mj_copyData(reference, model_, data_);
    for (int i = 0; i < 200; ++i) {
        mj_kdl::step(&robot_);
        mj_step(model_, reference);
    }
    EXPECT_EQ(std::memcmp(data_->qpos, reference->qpos, sizeof(mjtNum) * model_->nq), 0);
    EXPECT_EQ(std::memcmp(data_->qvel, reference->qvel, sizeof(mjtNum) * model_->nv), 0);
    mj_deleteData(reference);
}

TEST_F(SceneStateTest, StepHonoursAQposWrittenBetweenSteps)
{
    if (!model_) return;
    const int adr = mj_kdl::bind_scene_free_body(&scene_, "cube")->qpos_adr;
    mj_kdl::step(&robot_);
    mjData *reference = mj_makeData(model_);
    mj_copyData(reference, model_, data_);

    data_->qpos[adr + 2] = reference->qpos[adr + 2] = 2.0;
    mj_kdl::step(&robot_);
    mj_step(model_, reference);
    EXPECT_EQ(data_->qpos[adr + 2], reference->qpos[adr + 2]);

    KDL::Frame cube;
    ASSERT_TRUE(mj_kdl::get_body_frame(model_, data_, "cube", &cube));
    EXPECT_NEAR(cube.p.z(), data_->qpos[adr + 2], 1e-12);
    mj_deleteData(reference);
}

TEST_F(SceneStateTest, BindFreeBodyRejectsFixedUnknownAndDuplicate)
{
    if (!model_) return;
    const auto level = mj_kdl::get_log_level();
    mj_kdl::set_log_level(mj_kdl::LogLevel::NONE);

    EXPECT_EQ(mj_kdl::bind_scene_free_body(&scene_, "block"), nullptr);
    EXPECT_EQ(mj_kdl::bind_scene_free_body(&scene_, "no_such_body"), nullptr);
    ASSERT_NE(mj_kdl::bind_scene_free_body(&scene_, "cube"), nullptr);
    EXPECT_EQ(mj_kdl::bind_scene_free_body(&scene_, "cube"), nullptr);

    mj_kdl::set_log_level(level);
}

TEST_F(SceneStateTest, SceneJointTracksQposAndRejectsAFreeJoint)
{
    mj_kdl::SceneJointSlot *slot = mj_kdl::bind_scene_joint(&scene_, "joint_4");
    ASSERT_NE(slot, nullptr);

    KDL::JntArray q(robot_.n_joints);
    for (int i = 0; i < robot_.n_joints; ++i) q(i) = 0.0;
    q(3) = -1.25;
    mj_kdl::set_joint_pos(&robot_, q);
    mj_kdl::read_scene_state(&scene_, data_);
    EXPECT_NEAR(slot->position, -1.25, 1e-12);
    EXPECT_EQ(slot->velocity, data_->qvel[slot->dof_adr]);

    const auto level = mj_kdl::get_log_level();
    mj_kdl::set_log_level(mj_kdl::LogLevel::NONE);
    EXPECT_EQ(mj_kdl::bind_scene_joint(&scene_, "cube_free"), nullptr);
    mj_kdl::set_log_level(level);
}

TEST_F(SceneStateTest, ASlotAddressSurvivesFurtherBinds)
{
    mj_kdl::SceneWrenchSlot *first = mj_kdl::bind_scene_wrench(&scene_, "cube");
    ASSERT_NE(first, nullptr);

    for (int i = 0; i < kFillerBodies; ++i) {
        ASSERT_NE(
          mj_kdl::bind_scene_wrench(&scene_, ("filler_" + std::to_string(i)).c_str()), nullptr
        );
    }

    EXPECT_EQ(first, &scene_.wrenches.front());
    EXPECT_EQ(first->name, "cube");
    EXPECT_EQ(scene_.wrenches.size(), static_cast<std::size_t>(kFillerBodies) + 1);
}

TEST_F(SceneStateTest, ReadAndApplyTogetherMatchUpdate)
{
    if (!model_) return;
    robot_.ctrl_mode = mj_kdl::CtrlMode::TORQUE;
    for (int i = 0; i < robot_.n_joints; ++i) robot_.jnt_trq_cmd[i] = 0.1 * (i + 1);
    mj_kdl::step(&robot_);

    mj_kdl::update(&robot_);
    const std::vector<double> ctrl_after_update(data_->ctrl, data_->ctrl + model_->nu);
    const std::vector<double> qfrc_after_update(
      data_->qfrc_applied, data_->qfrc_applied + model_->nv
    );
    const std::vector<double> pos_after_update = robot_.jnt_pos_msr;

    std::fill(data_->ctrl, data_->ctrl + model_->nu, 0.0);
    std::fill(data_->qfrc_applied, data_->qfrc_applied + model_->nv, 0.0);

    mj_kdl::read_measurements(&robot_);
    mj_kdl::apply_commands(&robot_);
    EXPECT_EQ(std::vector<double>(data_->ctrl, data_->ctrl + model_->nu), ctrl_after_update);
    EXPECT_EQ(
      std::vector<double>(data_->qfrc_applied, data_->qfrc_applied + model_->nv), qfrc_after_update
    );
    EXPECT_EQ(robot_.jnt_pos_msr, pos_after_update);
}

TEST_F(SceneStateTest, ApplyClearsAWrenchThatIsNoLongerPushed)
{
    mj_kdl::SceneWrenchSlot *slot = mj_kdl::bind_scene_wrench(&scene_, "cube");
    ASSERT_NE(slot, nullptr);

    slot->wrench = KDL::Wrench(KDL::Vector(1.0, -2.0, 3.0), KDL::Vector(0.4, 0.5, 0.6));
    mj_kdl::apply_scene_state(&scene_, data_);
    const double *applied = data_->xfrc_applied + 6 * slot->body_id;
    EXPECT_EQ(applied[0], 1.0);
    EXPECT_EQ(applied[5], 0.6);

    slot->wrench = KDL::Wrench::Zero();
    mj_kdl::apply_scene_state(&scene_, data_);
    for (int k = 0; k < 6; ++k) EXPECT_EQ(applied[k], 0.0) << "component " << k;
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
