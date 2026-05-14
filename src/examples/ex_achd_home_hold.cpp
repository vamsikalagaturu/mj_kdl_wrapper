#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include "chainhdsolver_vereshchagin_fixed_joint.hpp"
#include <kdl/chainfksolverpos_recursive.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };
static constexpr double kKpLin       = 10.0;
static constexpr double kKiLin       = 5.0;
static constexpr double kKdLin       = 0.1;
static constexpr double kKpRot       = 10.0;
static constexpr double kKiRot       = 5.0;
static constexpr double kKdRot       = 0.1;
static constexpr double kKpPosture   = 0.0;
static constexpr double kKdPosture   = 0.05;
static constexpr double kPostureMax  = 0.5;
static constexpr double kTauMax      = 59.0;
static constexpr double kIntegralMax = 0.5;
static constexpr double kBetaMax     = 4.0;

static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path().parent_path(); }
static double clamp_abs(double v, double limit) { return std::max(-limit, std::min(limit, v)); }

static void fill_q_state(const mj_kdl::Robot &robot, KDL::JntArray &q, KDL::JntArray &qd)
{
    for (unsigned i = 0; i < q.rows(); ++i) {
        q(i)  = robot.jnt_pos_msr[i];
        qd(i) = robot.jnt_vel_msr[i];
    }
}

static void set_alpha_identity(KDL::Jacobian &alpha)
{
    for (unsigned i = 0; i < 6; ++i) {
        alpha.setColumn(i, KDL::Twist(KDL::Vector(i == 0, i == 1, i == 2),
                                      KDL::Vector(i == 3, i == 4, i == 5)));
    }
}

static bool achd_home_step(
  mj_kdl::Robot                               &robot,
  KDL::ChainFkSolverPos_recursive             &fk_pos,
  KDL::ChainHdSolver_Vereshchagin_Fixed_Joint &achd,
  const KDL::Frame                            &target,
  const KDL::JntArray                         &q_home,
  KDL::JntArray                               &q,
  KDL::JntArray                               &qd,
  KDL::JntArray                               &qdd,
  KDL::Jacobian                               &alpha,
  KDL::JntArray                               &beta,
  KDL::Wrenches                               &f_ext,
  KDL::JntArray                               &ff_tau,
  KDL::JntArray                               &tau,
  std::array<double, 6>                       &err_i,
  std::array<double, 6>                       &err_prev,
  bool                                        &first_pid
)
{
    fill_q_state(robot, q, qd);

    KDL::Frame current;
    fk_pos.JntToCart(q, current);

    const KDL::Twist err = KDL::diff(target, current);
    const double dt      = robot.model ? robot.model->opt.timestep : 0.002;

    const double e[6] = { err.vel.x(), err.vel.y(), err.vel.z(), err.rot.x(), err.rot.y(), err.rot.z() };
    if (first_pid) {
        for (unsigned i = 0; i < 6; ++i) err_prev[i] = e[i];
        first_pid = false;
    }
    for (unsigned i = 0; i < 6; ++i) err_i[i] = clamp_abs(err_i[i] + e[i] * dt, kIntegralMax);
    const double de[6] = {
        (e[0] - err_prev[0]) / dt,
        (e[1] - err_prev[1]) / dt,
        (e[2] - err_prev[2]) / dt,
        (e[3] - err_prev[3]) / dt,
        (e[4] - err_prev[4]) / dt,
        (e[5] - err_prev[5]) / dt,
    };

    beta(0) = clamp_abs(kKpLin * e[0] + kKiLin * err_i[0] + kKdLin * de[0], kBetaMax);
    beta(1) = clamp_abs(kKpLin * e[1] + kKiLin * err_i[1] + kKdLin * de[1], kBetaMax);
    beta(2) = clamp_abs(kKpLin * e[2] + kKiLin * err_i[2] + kKdLin * de[2], kBetaMax);
    beta(3) = clamp_abs(kKpRot * e[3] + kKiRot * err_i[3] + kKdRot * de[3], kBetaMax);
    beta(4) = clamp_abs(kKpRot * e[4] + kKiRot * err_i[4] + kKdRot * de[4], kBetaMax);
    beta(5) = clamp_abs(kKpRot * e[5] + kKiRot * err_i[5] + kKdRot * de[5], kBetaMax);
    for (unsigned i = 0; i < 6; ++i) err_prev[i] = e[i];

    for (unsigned i = 0; i < q.rows(); ++i) {
        ff_tau(i) = clamp_abs(kKpPosture * (q_home(i) - q(i)) - kKdPosture * qd(i), kPostureMax);
    }
    if (achd.CartToJnt(q, qd, qdd, alpha, beta, f_ext, ff_tau, tau) < 0) return false;

    for (unsigned i = 0; i < q.rows(); ++i) robot.jnt_trq_cmd[i] = clamp_abs(tau(i), kTauMax);
    return true;
}

int main(int argc, char **argv)
{
    bool headless = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--headless") headless = true;
    }

    const fs::path root = repo_root();
    const std::string arm_mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
    const std::string gripper_mjcf = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();

    mj_kdl::AttachmentSpec gripper;
    gripper.mjcf_path = gripper_mjcf.c_str();
    gripper.attach_to = "bracelet_link";
    gripper.prefix    = "g_";
    gripper.pos[2]    = -0.061525;
    gripper.euler[0]  = 180.0;

    mj_kdl::RobotSpec robot_spec;
    robot_spec.path = arm_mjcf.c_str();
    robot_spec.attachments.push_back(gripper);

    mj_kdl::SceneSpec scene;
    scene.robots.push_back(robot_spec);

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    if (!mj_kdl::build_scene(&model, &data, &scene)) return 1;

    mj_kdl::ToolFrameSpec tool;
    tool.tool_body = "g_base";
    tool.tcp_site  = "g_pinch";

    mj_kdl::Robot robot;
    if (!mj_kdl::init_robot_from_mjcf(&robot, model, data, "base_link", "bracelet_link", "", &tool)) {
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    const unsigned n  = robot.chain.getNrOfJoints();
    const unsigned ns = robot.chain.getNrOfSegments();

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

    mj_kdl::Env env;
    env.spec  = scene;
    env.model = model;
    env.data  = data;
    mj_kdl::env_add_robot(&env, &robot);
    env.on_reset = [&](mj_kdl::ResetContext *) { mj_kdl::set_joint_pos(&robot, q_home, false); };
    mj_kdl::reset(&env);

    KDL::ChainFkSolverPos_recursive fk_pos(robot.chain);
    KDL::Frame target;
    fk_pos.JntToCart(q_home, target);

    KDL::Twist root_acc(KDL::Vector(0.0, 0.0, -scene.gravity_z), KDL::Vector::Zero());
    KDL::ChainHdSolver_Vereshchagin_Fixed_Joint achd(robot.chain, root_acc, 6);

    KDL::JntArray q(n), qd(n), qdd(n), beta(6), ff_tau(n), tau(n);
    KDL::Wrenches f_ext(ns, KDL::Wrench::Zero());
    KDL::Jacobian alpha(6);
    std::array<double, 6> err_i{};
    set_alpha_identity(alpha);

    auto control_step = [&]() {
        mj_kdl::update(&robot);
        static std::array<double, 6> err_prev{};
        static bool first_pid = true;
        bool ok = achd_home_step(
          robot,
          fk_pos,
          achd,
          target,
          q_home,
          q,
          qd,
          qdd,
          alpha,
          beta,
          f_ext,
          ff_tau,
          tau,
          err_i,
          err_prev,
          first_pid
        );
        mj_kdl::update(&robot);
        return ok;
    };

    if (headless) {
        for (int i = 0; i < 2500; ++i) {
            if (!control_step() || !mj_kdl::step(&robot)) break;
        }
        mj_kdl::update(&robot);
        fill_q_state(robot, q, qd);
        KDL::Frame current;
        fk_pos.JntToCart(q, current);
        KDL::Twist err = KDL::diff(target, current);
        double max_qvel = 0.0;
        double max_qerr = 0.0;
        double max_qdd  = 0.0;
        double max_tau  = 0.0;
        double max_ff   = 0.0;
        for (unsigned i = 0; i < n; ++i) {
            max_qvel = std::max(max_qvel, std::abs(qd(i)));
            max_qerr = std::max(max_qerr, std::abs(q_home(i) - q(i)));
            max_qdd  = std::max(max_qdd, std::abs(qdd(i)));
            max_tau  = std::max(max_tau, std::abs(tau(i)));
            max_ff   = std::max(max_ff, std::abs(ff_tau(i)));
        }
        std::cout << "tcp_pos_err_mm=" << std::fixed << std::setprecision(3) << err.vel.Norm() * 1000.0
                  << " tcp_rot_err_rad=" << err.rot.Norm()
                  << " max_qerr=" << max_qerr
                  << " max_qvel=" << max_qvel
                  << " max_qdd=" << max_qdd
                  << " max_constraint_tau=" << max_tau
                  << " max_ff_tau=" << max_ff << "\n";
    } else {
        mj_kdl::Viewer viewer;
        if (!mj_kdl::init_window_sim(&viewer, &robot)) return 1;
        double prev_time = data->time;
        while (true) {
            if (data->time < prev_time - 1e-6) {
                mj_kdl::reset(&env);
                err_i = {};
            }
            prev_time = data->time;
            if (!control_step() || !mj_kdl::step(&robot)) break;
        }
        mj_kdl::cleanup(&viewer);
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
