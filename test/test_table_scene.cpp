/* test_table_scene.cpp
 * Load a robot arm on a table with pickable objects (cubes and spheres).
 * Runs KDL gravity compensation so the arm holds position.
 * Also tests runtime scene_add_object / scene_remove_object.
 * Self-skips when Menagerie is absent. */

#include <gtest/gtest.h>

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "example_paths.hpp"

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chaindynparam.hpp>

#include <memory>
#include <string>
#include <filesystem>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

namespace fs = std::filesystem;
// Free-jointed primitives must stay world-anchored (MuJoCo: freejoint only on
// top level), so their z is surface_z + half_height. The robot, which has no
// freejoint at its root, attaches to the table via a site instead.
static constexpr double kCubeMass         = 0.1; // [kg]
static constexpr double kSphereMass       = 0.05;
static constexpr double kPickFriction[3]  = { 0.8, 0.02, 0.001 };

TEST(SceneObjectTransform, PathBackedObjectAppliesQuat)
{
    std::string       table_mjcf = mj_kdl_examples::find_asset("table.xml");
    mj_kdl::SceneSpec spec;
    spec.timestep = 0.002;
    // extrinsic XYZ euler (30, 40, 50) deg
    spec.objects.push_back({
      .name      = "turned",
      .mjcf_path = table_mjcf,
      .quat = { 0.08080468869083995, 0.40219849353410964, 0.30337177447125957, 0.860042173697679 },
      .fixed     = true,
    });

    mj_kdl::Env env;
    ASSERT_TRUE(mj_kdl::init_env(&env, &spec));
    KDL::Frame frame;
    ASSERT_TRUE(mj_kdl::get_site_frame(&env, "turned_table_top", &frame));
    const KDL::Vector y = frame.M * KDL::Vector(0.0, 1.0, 0.0);
    EXPECT_NEAR(y.x(), -0.456825992585671, 1e-9);
    EXPECT_NEAR(y.y(), 0.802872337479472, 1e-9);
    EXPECT_NEAR(y.z(), 0.383022221559489, 1e-9);
}

static mj_kdl::SceneObject make_box(
  const char *name, double x, double y, double hx, double hy, double hz,
  float r, float g, float b, double surface_z
)
{
    return mj_kdl::SceneObject{
        .name      = name,
        .mjcf_path = "",
        .shape     = mj_kdl::Shape::BOX,
        .size      = { hx, hy, hz },
        .pos       = { x, y, surface_z + hz },
        .rgba      = { r, g, b, 1.0f },
        .mass      = kCubeMass,
        .friction  = { kPickFriction[0], kPickFriction[1], kPickFriction[2] },
    };
}

static mj_kdl::SceneObject make_sphere(
  const char *name, double x, double y, double radius,
  float r, float g, float b, double surface_z
)
{
    return mj_kdl::SceneObject{
        .name      = name,
        .mjcf_path = "",
        .shape     = mj_kdl::Shape::SPHERE,
        .size      = { radius, 0.0, 0.0 },
        .pos       = { x, y, surface_z + radius },
        .rgba      = { r, g, b, 1.0f },
        .mass      = kSphereMass,
        .friction  = { kPickFriction[0], kPickFriction[1], kPickFriction[2] },
    };
}

class TableSceneTest : public testing::Test
{
  protected:
    std::string       mjcf_;
    mj_kdl::SceneSpec spec_;
    mj_kdl::SceneObject table_obj_;
    std::string       table_mount_site_; // compiled site name; lifetime backs RobotSpec.attach_to.name
    mj_kdl::Env       env_;
    mj_kdl::Robot     s_;

    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;
    std::unique_ptr<KDL::ChainDynParam>              dyn_;

    unsigned      n_ = 0;
    KDL::JntArray q_home_;

    void SetUp() override
    {
        mjcf_ = mj_kdl_examples::find_menagerie_model("kinova_gen3/gen3.xml");
        if (!fs::exists(mjcf_)) {
            GTEST_SKIP() << mjcf_ << " not found";
            return;
        }
        std::string table_mjcf = mj_kdl_examples::find_asset("table.xml");

        // Table asset origin is the tabletop surface center; only the table
        // itself carries a world-frame z. Robot and objects derive their
        // height from attach_to.
        const double surface_z = 0.7;

        table_obj_ = {
            .name      = "table",
            .mjcf_path = table_mjcf,
            .pos       = { 0.0, 0.0, surface_z },
            .fixed     = true,
        };
        table_mount_site_ = mj_kdl::scene_object_site_name(table_obj_, "table_top");

        std::vector<mj_kdl::SceneObject> objects;
        objects.push_back(table_obj_);
        objects.push_back(make_box("red_cube",     0.35,  0.10, 0.03, 0.03, 0.03, 1.0f, 0.2f, 0.2f, surface_z));
        objects.push_back(make_box("green_cube",   0.35, -0.10, 0.03, 0.03, 0.03, 0.2f, 1.0f, 0.2f, surface_z));
        objects.push_back(make_box("blue_cube",    0.35,  0.30, 0.04, 0.04, 0.04, 0.2f, 0.2f, 1.0f, surface_z));
        objects.push_back(make_sphere("orange_sphere", -0.20,  0.20, 0.035, 1.0f, 0.55f, 0.0f, surface_z));
        objects.push_back(make_sphere("purple_sphere", -0.20, -0.20, 0.025, 0.7f, 0.0f,  0.9f, surface_z));

        spec_.objects    = objects;
        spec_.timestep   = 0.002;
        spec_.gravity_z  = -9.81;
        spec_.add_floor  = true;
        spec_.add_skybox = true;

        spec_.robots.push_back(mj_kdl::RobotSpec{
            .path      = mjcf_,
            .attach_to = { mj_kdl::AttachKind::Site, table_mount_site_ },
            .attachments = {},
        });

        ASSERT_TRUE(mj_kdl::init_env(&env_, &spec_));
        KDL::Frame world_T_table_top;
        ASSERT_TRUE(mj_kdl::get_site_frame(&env_, table_mount_site_.c_str(), &world_T_table_top));
        EXPECT_NEAR(world_T_table_top.p.z(), surface_z, 1e-9);

        ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&s_, &env_, "base_link", "bracelet_link"));

        n_ = static_cast<unsigned>(s_.n_joints);

        fk_  = std::make_unique<KDL::ChainFkSolverPos_recursive>(s_.chain);
        dyn_ = std::make_unique<KDL::ChainDynParam>(s_.chain, KDL::Vector(0, 0, spec_.gravity_z));

        q_home_.resize(n_);
        for (unsigned i = 0; i < n_; ++i) q_home_(i) = kHomePose[i];
        mj_kdl::set_joint_pos(&s_, q_home_);
    }
};

TEST_F(TableSceneTest, GravityCompDrift)
{
    KDL::Frame ee_init;
    fk_->JntToCart(q_home_, ee_init);

    s_.ctrl_mode = mj_kdl::CtrlMode::TORQUE;
    KDL::JntArray q(n_), g(n_);
    // Prime jnt_trq_cmd so the first update() applies compensation immediately.
    dyn_->JntToGravity(q_home_, g);
    for (unsigned j = 0; j < n_; ++j) s_.jnt_trq_cmd[j] = g(j);
    for (int i = 0; i < 500; ++i) {
        mj_kdl::update(&env_);
        for (unsigned j = 0; j < n_; ++j) q(j) = s_.jnt_pos_msr[j];
        dyn_->JntToGravity(q, g);
        for (unsigned j = 0; j < n_; ++j) s_.jnt_trq_cmd[j] = g(j);
        mj_kdl::step(&env_);
    }

    KDL::JntArray q_end(n_);
    KDL::Frame    ee_end;
    for (unsigned j = 0; j < n_; ++j) q_end(j) = s_.jnt_pos_msr[j];
    fk_->JntToCart(q_end, ee_end);
    double drift = (ee_init.p - ee_end.p).Norm();

    ASSERT_LE(drift, 0.001) << "drift " << drift * 1000.0 << " mm exceeds 1 mm threshold";
}

TEST_F(TableSceneTest, EnvAddRemoveReinitsRobot)
{
    const int nq_before = env_.model->nq;

    // Objects are compiled ahead of the robot, so each rebuild moves every robot address.
    auto expect_maps_match_model = [&] {
        for (int i = 0; i < s_.n_joints; ++i) {
            const int jid = mj_name2id(env_.model, mjOBJ_JOINT, s_.joint_names[i].c_str());
            ASSERT_GE(jid, 0);
            env_.data->qpos[env_.model->jnt_qposadr[jid]] = 0.1 * (i + 1);
        }
        mj_kdl::update(&env_);
        for (int i = 0; i < s_.n_joints; ++i) EXPECT_DOUBLE_EQ(s_.jnt_pos_msr[i], 0.1 * (i + 1));
    };

    mj_kdl::SceneFreeBodySlot *red = mj_kdl::bind_scene_free_body(&env_.scene, "red_cube");
    ASSERT_NE(red, nullptr);

    KDL::Frame yellow_frame;
    EXPECT_FALSE(mj_kdl::get_body_frame(&env_, "yellow_cube", &yellow_frame));

    const double        surface_z = 0.7;
    mj_kdl::SceneObject extra =
      make_box("yellow_cube", 0.0, 0.4, 0.03, 0.03, 0.03, 1.0f, 1.0f, 0.0f, surface_z);

    ASSERT_TRUE(mj_kdl::scene_add_object(&env_, extra)) << "Env scene_add_object() failed";
    EXPECT_GT(env_.model->nq, nq_before) << "nq should grow after adding a free object";
    EXPECT_EQ(s_.n_joints, 7) << "robot chain should still have 7 joints after rebuild";
    EXPECT_EQ(s_.model, env_.model) << "robot model pointer should be updated";
    expect_maps_match_model();
    // A name looked up on the freed model must not answer for the new one at the same address.
    EXPECT_TRUE(mj_kdl::get_body_frame(&env_, "yellow_cube", &yellow_frame));

    mj_kdl::SceneFreeBodySlot *yellow = mj_kdl::bind_scene_free_body(&env_.scene, "yellow_cube");
    ASSERT_NE(yellow, nullptr);

    // The object is removed even though a slot bound to it goes unbound with it.
    EXPECT_TRUE(mj_kdl::scene_remove_object(&env_, "yellow_cube"));
    EXPECT_EQ(env_.model->nq, nq_before) << "nq should return to original after removal";
    EXPECT_EQ(s_.n_joints, 7) << "robot chain should still have 7 joints after removal";
    expect_maps_match_model();

    EXPECT_EQ(yellow->qpos_adr, -1) << "yellow_cube is gone";
    const int red_bid = mj_name2id(env_.model, mjOBJ_BODY, "red_cube");
    EXPECT_EQ(red->qpos_adr, env_.model->jnt_qposadr[env_.model->body_jntadr[red_bid]]);
    mj_kdl::update(&env_);
    EXPECT_GT(red->seq, 0u);
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
