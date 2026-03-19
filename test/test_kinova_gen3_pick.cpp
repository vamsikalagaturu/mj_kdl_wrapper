/* test_kinova_gen3_pick.cpp
 * Kinova Gen3 + Robotiq 2F-85: pick a cube from the floor.
 *
 * Cube (4 cm, orange) spawned at (0.4, 0, 0.02) in front of arm.
 * Target orientation: KDL::Rotation::Identity() on bracelet_link, which
 * makes the gripper approach axis point straight down (gripper is attached
 * 180° around X below bracelet_link).
 *
 * Bracelet_link Z target = cube_center_Z + 0.184
 *   (0.062 gs_offset + 0.122 finger reach along gripper Z axis)
 *
 * Tests:
 *   1. KDL chain: 7 arm joints.
 *   2. IK converges for pre-grasp, grasp, lift (pos error < 2 mm).
 *   3. Headless pick simulation (10.5 s): cube lifted > 0.20 m. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <tinyxml2.h>

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

/* Bracelet_link → finger pad offset along gripper Z (measured from 2F-85 geometry):
 *   gs_offset(0.0615) + base_mount(0.007) + base(0.0038) +
 *   spring_link_z(0.0609) + follower_z(0.0375) + pad_z(0.01352) = 0.18422 m */
static constexpr double kGripperReach = 0.18422;

/* Cube spawn */
static constexpr double kCubeX  = 0.4;
static constexpr double kCubeY  = 0.0;
static constexpr double kCubeZ  = 0.02; // centre (bottom at z=0)
static constexpr double kCubeHS = 0.02; // half-size: 4 cm cube

static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

/* Add contact exclusions between bracelet_link and all g_* gripper bodies. */
static bool patch_contact_exclusions(const std::string &path)
{
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) return false;
    auto *root = doc.FirstChildElement("mujoco");
    if (!root) return false;

    auto *contact = root->FirstChildElement("contact");
    if (!contact) {
        contact = doc.NewElement("contact");
        root->InsertEndChild(contact);
    }

    const char *gripper_bodies[] = { "g_base_mount",
        "g_base",
        "g_left_driver",
        "g_right_driver",
        "g_left_spring_link",
        "g_right_spring_link",
        "g_left_follower",
        "g_right_follower",
        "g_left_coupler",
        "g_right_coupler",
        "g_left_pad",
        "g_right_pad",
        "g_left_silicone_pad",
        "g_right_silicone_pad" };
    for (const char *gb : gripper_bodies) {
        auto *exc = doc.NewElement("exclude");
        exc->SetAttribute("body1", "bracelet_link");
        exc->SetAttribute("body2", gb);
        contact->InsertEndChild(exc);
    }
    return doc.SaveFile(path.c_str()) == tinyxml2::XML_SUCCESS;
}

static double clamp01(double t) { return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t); }

static void
  lerp_q(const KDL::JntArray &from, const KDL::JntArray &to, double alpha, KDL::JntArray &out)
{
    unsigned n = from.rows();
    out.resize(n);
    for (unsigned i = 0; i < n; ++i) out(i) = from(i) + alpha * (to(i) - from(i));
}

struct Phase
{
    double               t_start, duration;
    const KDL::JntArray *q_from;
    const KDL::JntArray *q_to;
    double               gripper;
};

class PickTest : public ::testing::Test
{
  protected:
    fs::path                                         root_;
    std::string                                      combined_;
    mjModel                                         *model_ = nullptr;
    mjData                                          *data_  = nullptr;
    mj_kdl::Robot                                    s_;
    int                                              fingers_act_ = -1;
    int                                              cube_bid_    = -1;
    int                                              cube_jnt_    = -1;
    unsigned                                         n_           = 0;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;
    std::unique_ptr<KDL::ChainIkSolverVel_pinv>      ik_vel_;
    std::unique_ptr<KDL::ChainIkSolverPos_NR_JL>     ik_;
    KDL::JntArray                                    q_home_kdl_;
    KDL::JntArray                                    q_pregrasp_, q_grasp_, q_lift_;
    int                                              key_id_ = -1;

    void SetUp() override
    {
        root_ = repo_root();
        if (!fs::exists(root_ / "third_party/menagerie")) {
            GTEST_SKIP() << "third_party/menagerie/ not found — run locally with the submodule";
            return;
        }

        const std::string arm_mjcf =
          (root_ / "third_party/menagerie/kinova_gen3/gen3.xml").string();
        const std::string grp_mjcf =
          (root_ / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
        combined_ = "/tmp/gen3_with_2f85_pick.xml";

        mj_kdl::GripperSpec gs;
        gs.mjcf_path = grp_mjcf.c_str();
        gs.attach_to = "bracelet_link";
        gs.prefix    = "g_";
        gs.pos[0]    = 0.0;
        gs.pos[1]    = 0.0;
        gs.pos[2]    = -0.061525;
        gs.quat[0]   = 0.0;
        gs.quat[1]   = 1.0;
        gs.quat[2]   = 0.0;
        gs.quat[3]   = 0.0;

        ASSERT_TRUE(mj_kdl::attach_gripper(arm_mjcf.c_str(), &gs, combined_.c_str()))
          << "attach_gripper() returned false";
        ASSERT_TRUE(mj_kdl::patch_mjcf_add_skybox(combined_.c_str()))
          << "patch_mjcf_add_skybox() returned false";
        ASSERT_TRUE(mj_kdl::patch_mjcf_add_floor(combined_.c_str()))
          << "patch_mjcf_add_floor() returned false";
        ASSERT_TRUE(patch_contact_exclusions(combined_))
          << "patch_contact_exclusions() returned false";

        mj_kdl::SceneObject cube_obj;
        cube_obj.name        = "cube";
        cube_obj.shape       = mj_kdl::ObjShape::BOX;
        cube_obj.size[0]     = kCubeHS;
        cube_obj.size[1]     = kCubeHS;
        cube_obj.size[2]     = kCubeHS;
        cube_obj.pos[0]      = kCubeX;
        cube_obj.pos[1]      = kCubeY;
        cube_obj.pos[2]      = kCubeZ;
        cube_obj.rgba[0]     = 1.0f;
        cube_obj.rgba[1]     = 0.5f;
        cube_obj.rgba[2]     = 0.0f;
        cube_obj.rgba[3]     = 1.0f;
        cube_obj.mass        = 0.1;
        cube_obj.condim      = 4;
        cube_obj.friction[0] = 0.8;
        cube_obj.friction[1] = 0.02;
        cube_obj.friction[2] = 0.001;

        ASSERT_TRUE(mj_kdl::patch_mjcf_add_objects(combined_.c_str(), { cube_obj }))
          << "patch_mjcf_add_objects() returned false";

        ASSERT_TRUE(mj_kdl::load_mjcf(&model_, &data_, combined_.c_str()))
          << "load_mjcf() returned false for combined MJCF";

        cube_bid_ = mj_name2id(model_, mjOBJ_BODY, "cube");
        cube_jnt_ = mj_name2id(model_, mjOBJ_JOINT, "cube_joint");
        ASSERT_GE(cube_bid_, 0) << "cube body not found";
        ASSERT_GE(cube_jnt_, 0) << "cube_joint not found";

        TEST_INFO("model: nq=" << model_->nq << " nbody=" << model_->nbody);

        ASSERT_TRUE(mj_kdl::init_from_mjcf(&s_, model_, data_, "base_link", "bracelet_link"))
          << "init_from_mjcf() returned false";
        n_ = s_.chain.getNrOfJoints();
        ASSERT_EQ(n_, 7u);

        fk_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(s_.chain);

        KDL::JntArray q_min(n_), q_max(n_);
        for (unsigned i = 0; i < n_; ++i) {
            int jid = model_->dof_jntid[s_.kdl_to_mj_dof[i]];
            if (model_->jnt_limited[jid]) {
                q_min(i) = model_->jnt_range[2 * jid];
                q_max(i) = model_->jnt_range[2 * jid + 1];
            } else {
                q_min(i) = -2 * M_PI;
                q_max(i) = 2 * M_PI;
            }
        }
        ik_vel_ = std::make_unique<KDL::ChainIkSolverVel_pinv>(s_.chain);
        ik_     = std::make_unique<KDL::ChainIkSolverPos_NR_JL>(
          s_.chain, q_min, q_max, *fk_, *ik_vel_, 500, 1e-5);

        fingers_act_ = mj_name2id(model_, mjOBJ_ACTUATOR, "g_fingers_actuator");
        ASSERT_GE(fingers_act_, 0) << "g_fingers_actuator not found";

        key_id_ = mj_name2id(model_, mjOBJ_KEY, "home");
        q_home_kdl_.resize(n_);
        for (unsigned i = 0; i < n_; ++i) q_home_kdl_(i) = kHomePose[i];

        /* Waypoint Z targets (using exact gripper reach geometry).
         * +0.02 m offset: keeps finger geometry clear of the floor when the gripper
         * is open during approach (open fingers extend further down than closed). */
        const double kGraspZ    = kCubeZ + kGripperReach + 0.02;
        const double kPreGraspZ = kGraspZ + 0.20;
        const double kLiftZ     = kGraspZ + 0.30;

        const KDL::Rotation kDownRot = KDL::Rotation::Identity();

        struct WP
        {
            const char          *name;
            double               z;
            KDL::JntArray       *out;
            const KDL::JntArray *seed;
        };
        q_pregrasp_.resize(n_);
        q_grasp_.resize(n_);
        q_lift_.resize(n_);
        WP wps[] = {
            { "pre-grasp", kPreGraspZ, &q_pregrasp_, &q_home_kdl_ },
            { "grasp", kGraspZ, &q_grasp_, &q_pregrasp_ },
            { "lift", kLiftZ, &q_lift_, &q_grasp_ },
        };

        for (auto &wp : wps) {
            KDL::Frame target(kDownRot, KDL::Vector(kCubeX, kCubeY, wp.z));
            int        ret = ik_->CartToJnt(*wp.seed, target, *wp.out);
            KDL::Frame ik_frame;
            fk_->JntToCart(*wp.out, ik_frame);
            double err = (ik_frame.p - target.p).Norm() * 1000.0;
            ASSERT_GE(ret, 0) << "IK SetUp " << wp.name << " failed (ret=" << ret << ")";
            ASSERT_LE(err, 2.0) << "IK SetUp " << wp.name << " pos_err=" << err << " mm > 2 mm";
        }
    }

    void TearDown() override
    {
        if (model_) {
            mj_kdl::cleanup(&s_);
            mj_kdl::destroy_scene(model_, data_);
        }
    }

    /* Place cube at spawn position. */
    void reset_cube(mjData *d)
    {
        int qadr          = model_->jnt_qposadr[cube_jnt_];
        d->qpos[qadr + 0] = kCubeX;
        d->qpos[qadr + 1] = kCubeY;
        d->qpos[qadr + 2] = kCubeZ;
        d->qpos[qadr + 3] = 1.0;
        d->qpos[qadr + 4] = d->qpos[qadr + 5] = d->qpos[qadr + 6] = 0.0;
    }

    /* Apply scripted position+gravity-comp control for one simulation step. */
    void apply_control(mjModel * /*m*/, mjData *d, const Phase *phases, int n_phases)
    {
        double       t  = d->time;
        const Phase *ph = &phases[n_phases - 1];
        for (int pi = 0; pi < n_phases - 1; ++pi)
            if (t < phases[pi + 1].t_start) {
                ph = &phases[pi];
                break;
            }

        double        alpha = clamp01((t - ph->t_start) / ph->duration);
        KDL::JntArray q_target(n_);
        lerp_q(*ph->q_from, *ph->q_to, alpha, q_target);

        for (unsigned i = 0; i < n_; ++i) {
            int dof = s_.kdl_to_mj_dof[i];
            /* Position actuators track q_target (stable: force clamped to forcerange).
             * qfrc_applied adds gravity compensation so the actuator P-term
             * only needs to handle tracking error, not fight gravity. */
            d->ctrl[i]           = q_target(i);
            d->qfrc_applied[dof] = d->qfrc_bias[dof];
        }
        d->ctrl[fingers_act_] = ph->gripper;
    }
};

TEST_F(PickTest, KDLChain)
{
    TEST_INFO("KDL chain: " << n_ << " joints");
    EXPECT_EQ(n_, 7u);
}

TEST_F(PickTest, IKConvergence)
{
    /* Waypoint Z targets */
    const double        kGraspZ    = kCubeZ + kGripperReach + 0.02;
    const double        kPreGraspZ = kGraspZ + 0.20;
    const double        kLiftZ     = kGraspZ + 0.30;
    const KDL::Rotation kDownRot   = KDL::Rotation::Identity();

    struct WP
    {
        const char *name;
        double      z;
    };
    WP wps[] = { { "pre-grasp", kPreGraspZ }, { "grasp", kGraspZ }, { "lift", kLiftZ } };

    KDL::JntArray        q_out(n_);
    const KDL::JntArray *seed = &q_home_kdl_;
    KDL::JntArray        q_prev(n_);

    for (int wi = 0; wi < 3; ++wi) {
        KDL::Frame target(kDownRot, KDL::Vector(kCubeX, kCubeY, wps[wi].z));
        int        ret = ik_->CartToJnt(*seed, target, q_out);

        KDL::Frame ik_frame;
        fk_->JntToCart(q_out, ik_frame);
        double err = (ik_frame.p - target.p).Norm() * 1000.0;

        TEST_INFO("IK " << wps[wi].name << " z=" << wps[wi].z << " pos_err=" << std::fixed
                        << std::setprecision(2) << err << " mm");

        EXPECT_GE(ret, 0) << "IK " << wps[wi].name << " solver returned error";
        EXPECT_LE(err, 2.0) << "IK " << wps[wi].name << " pos_err=" << err << " mm > 2 mm";

        q_prev = q_out;
        seed   = &q_prev;
    }
}

TEST_F(PickTest, CubeLifted)
{
    Phase phases[] = {
        { 0.0, 1.0, &q_home_kdl_, &q_home_kdl_, 0.0 }, // hold home
        { 1.0, 2.0, &q_home_kdl_, &q_pregrasp_, 0.0 }, // → pre-grasp
        { 3.0, 2.0, &q_pregrasp_, &q_grasp_, 0.0 },    // descend
        { 5.0, 1.5, &q_grasp_, &q_grasp_, 255.0 },     // close gripper
        { 6.5, 3.0, &q_grasp_, &q_lift_, 255.0 },      // lift
        { 9.5, 1e9, &q_lift_, &q_lift_, 255.0 },       // hold
    };
    constexpr int kNPhases = static_cast<int>(sizeof(phases) / sizeof(phases[0]));

    if (key_id_ >= 0)
        mj_resetDataKeyframe(model_, data_, key_id_);
    else
        mj_kdl::sync_from_kdl(&s_, q_home_kdl_);
    reset_cube(data_);
    mj_forward(model_, data_);

    const int kSteps = static_cast<int>(10.5 / model_->opt.timestep);
    for (int step = 0; step < kSteps; ++step) {
        apply_control(model_, data_, phases, kNPhases);
        mj_step(model_, data_);
    }

    int    qadr         = model_->jnt_qposadr[cube_jnt_];
    double cube_final_z = data_->qpos[qadr + 2];

    TEST_INFO("cube Z after pick simulation: " << std::fixed << std::setprecision(3) << cube_final_z
                                               << " m");

    EXPECT_GT(cube_final_z, 0.20) << "cube Z " << cube_final_z
                                  << " m < 0.20 m threshold (cube not lifted)";
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
