/* test_table_scene.cpp
 * Load a robot arm on a table with pickable objects (cubes and spheres).
 * Runs KDL gravity compensation so the arm holds position.
 * Also tests runtime scene_add_object / scene_remove_object. */

#include <gtest/gtest.h>

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "test_utils.hpp"

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chaindynparam.hpp>

#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <filesystem>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

static mj_kdl::SceneObject make_box(const char *name,
  double                                        x,
  double                                        y,
  double                                        hx,
  double                                        hy,
  double                                        hz,
  float                                         r,
  float                                         g,
  float                                         b,
  double                                        surface_z)
{
    mj_kdl::SceneObject o;
    o.name    = name;
    o.shape   = mj_kdl::ObjShape::BOX;
    o.size[0] = hx;
    o.size[1] = hy;
    o.size[2] = hz;
    o.pos[0]  = x;
    o.pos[1]  = y;
    o.pos[2]  = surface_z + hz;
    o.rgba[0] = r;
    o.rgba[1] = g;
    o.rgba[2] = b;
    o.rgba[3] = 1.0f;
    return o;
}

static mj_kdl::SceneObject make_sphere(const char *name,
  double                                           x,
  double                                           y,
  double                                           radius,
  float                                            r,
  float                                            g,
  float                                            b,
  double                                           surface_z)
{
    mj_kdl::SceneObject o;
    o.name    = name;
    o.shape   = mj_kdl::ObjShape::SPHERE;
    o.size[0] = radius;
    o.pos[0]  = x;
    o.pos[1]  = y;
    o.pos[2]  = surface_z + radius;
    o.rgba[0] = r;
    o.rgba[1] = g;
    o.rgba[2] = b;
    o.rgba[3] = 1.0f;
    return o;
}

class TableSceneTest : public ::testing::Test
{
  protected:
    std::string       urdf_;
    mj_kdl::SceneSpec spec_;
    mjModel          *model_ = nullptr;
    mjData           *data_  = nullptr;
    mj_kdl::Robot     s_;
    bool              s_cleaned_ = false;

    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;
    std::unique_ptr<KDL::ChainDynParam>              dyn_;

    unsigned      n_ = 0;
    KDL::JntArray q_home_;

    void SetUp() override
    {
        urdf_ = (repo_root() / "assets/gen3_urdf/GEN3_URDF_V12.urdf").string();

        /* Table setup: surface at z = 0.7 m, robot mounted at table centre. */
        mj_kdl::TableSpec table;
        table.enabled     = true;
        table.pos[0]      = 0.0;
        table.pos[1]      = 0.0;
        table.pos[2]      = 0.7; // surface height
        table.top_size[0] = 0.8; // half-extent x
        table.top_size[1] = 0.6; // half-extent y
        table.thickness   = 0.04;
        table.leg_radius  = 0.03;

        double surface_z = table.pos[2];

        std::vector<mj_kdl::SceneObject> objects;
        objects.push_back(
          make_box("red_cube", 0.35, 0.10, 0.03, 0.03, 0.03, 1.0f, 0.2f, 0.2f, surface_z));
        objects.push_back(
          make_box("green_cube", 0.35, -0.10, 0.03, 0.03, 0.03, 0.2f, 1.0f, 0.2f, surface_z));
        objects.push_back(
          make_box("blue_cube", 0.35, 0.30, 0.04, 0.04, 0.04, 0.2f, 0.2f, 1.0f, surface_z));
        objects.push_back(
          make_sphere("orange_sphere", -0.20, 0.20, 0.035, 1.0f, 0.55f, 0.0f, surface_z));
        objects.push_back(
          make_sphere("purple_sphere", -0.20, -0.20, 0.025, 0.7f, 0.0f, 0.9f, surface_z));

        spec_.table     = table;
        spec_.objects   = objects;
        spec_.add_floor = true;
        spec_.gravity_z = -9.81;

        mj_kdl::SceneRobot robot;
        robot.urdf_path = urdf_.c_str();
        robot.prefix    = "";
        robot.pos[0]    = 0.0;
        robot.pos[1]    = 0.0;
        robot.pos[2]    = surface_z;
        spec_.robots.push_back(robot);

        ASSERT_TRUE(mj_kdl::build_scene(&model_, &data_, &spec_));
        TEST_INFO(model_->nbody << " bodies, " << model_->nq << " DOFs");

        ASSERT_TRUE(
          mj_kdl::init_robot(&s_, model_, data_, urdf_.c_str(), "base_link", "EndEffector_Link"));
        TEST_INFO(s_.n_joints << " joints");

        n_ = static_cast<unsigned>(s_.n_joints);

        fk_  = std::make_unique<KDL::ChainFkSolverPos_recursive>(s_.chain);
        dyn_ = std::make_unique<KDL::ChainDynParam>(s_.chain, KDL::Vector(0, 0, spec_.gravity_z));

        q_home_.resize(n_);
        for (unsigned i = 0; i < n_; ++i) q_home_(i) = kHomePose[i];
        mj_kdl::sync_from_kdl(&s_, q_home_);
        mj_forward(model_, data_);
    }

    void TearDown() override
    {
        if (!s_cleaned_) mj_kdl::cleanup(&s_);
        if (model_) mj_kdl::destroy_scene(model_, data_);
    }
};

TEST_F(TableSceneTest, GravityCompDrift)
{
    KDL::Frame ee_init;
    fk_->JntToCart(q_home_, ee_init);

    for (int i = 0; i < 500; ++i) {
        KDL::JntArray q, g(n_);
        mj_kdl::sync_to_kdl(&s_, q);
        dyn_->JntToGravity(q, g);
        mj_kdl::set_torques(&s_, g);
        mj_kdl::step(&s_);
    }

    KDL::JntArray q_end;
    KDL::Frame    ee_end;
    mj_kdl::sync_to_kdl(&s_, q_end);
    fk_->JntToCart(q_end, ee_end);
    double drift = (ee_init.p - ee_end.p).Norm();

    TEST_INFO("EE drift after 500 steps: " << std::fixed << std::setprecision(3) << drift * 1000.0
                                           << " mm");

    ASSERT_LE(drift, 0.001) << "drift " << drift * 1000.0 << " mm exceeds 1 mm threshold";
}

TEST_F(TableSceneTest, AddRemoveObject)
{
    int nbody_before = model_->nbody;

    mj_kdl::cleanup(&s_);
    s_cleaned_ = true;

    double              surface_z = spec_.table.pos[2];
    mj_kdl::SceneObject extra =
      make_box("yellow_cube", 0.0, 0.4, 0.03, 0.03, 0.03, 1.0f, 1.0f, 0.0f, surface_z);

    ASSERT_TRUE(mj_kdl::scene_add_object(&model_, &data_, &spec_, extra))
      << "scene_add_object() returned false";
    TEST_INFO("bodies: " << nbody_before << " -> " << model_->nbody << " (after add)");

    ASSERT_TRUE(mj_kdl::scene_remove_object(&model_, &data_, &spec_, "yellow_cube"))
      << "scene_remove_object() returned false";
    TEST_INFO("bodies: " << model_->nbody << " (after remove)");
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
