// Press on the table with a commanded wrench, measured against the table's contact force.
// --variant weighted passes it through (w_f_ext = 1), main lets the constraint compensate it.
#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "example_paths.hpp"

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainhdsolver_vereshchagin.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace
{

constexpr double kHomePose[7]  = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };
constexpr double kTableZ       = 0.70;
double g_press_force = 10.0;  // [N] commanded, straight down at the tool; --force
constexpr double kDescendVel   = 0.08;  // [m/s]
constexpr double kContactForce = 1.0;   // [N] table normal force that ends the descent
constexpr double kHoldTime     = 1.0;   // [s] settle before descending, and before pressing
constexpr double kPressTime    = 4.0;   // [s] headless press duration
constexpr double kKpLin        = 200.0;
constexpr double kKdLin        = 30.0;
constexpr double kKpRot        = 175.0;
constexpr double kKdRot        = 28.0;
constexpr double kBetaMax      = 120.0;
constexpr double kTauMax       = 59.0;
constexpr const char *kGripperActuator = "g_fingers_actuator";
constexpr const char *kTableTopGeom    = "table_top";

enum class Variant { Weighted, Main };
enum class Phase { Hold, Descend, Settle, Press };

const char *phase_name(Phase p)
{
    switch (p) {
    case Phase::Hold: return "hold";
    case Phase::Descend: return "descend";
    case Phase::Settle: return "settle";
    case Phase::Press: return "press";
    }
    return "?";
}

double clamp_abs(double v, double limit) { return std::max(-limit, std::min(limit, v)); }

// Sum of contact normal forces on one geom; positive pushes the bodies apart.
double contact_normal_force(const mjModel *model, mjData *data, int geom)
{
    double total = 0.0;
    for (int i = 0; i < data->ncon; ++i) {
        const mjContact &c = data->contact[i];
        if (c.geom1 != geom && c.geom2 != geom) continue;
        double f[6];
        mj_contactForce(model, data, i, f);
        total += f[0];
    }
    return total;
}

void set_alpha(KDL::Jacobian &alpha, bool free_z)
{
    unsigned c = 0;
    alpha.setColumn(c++, KDL::Twist(KDL::Vector(1, 0, 0), KDL::Vector::Zero()));
    alpha.setColumn(c++, KDL::Twist(KDL::Vector(0, 1, 0), KDL::Vector::Zero()));
    if (!free_z) alpha.setColumn(c++, KDL::Twist(KDL::Vector(0, 0, 1), KDL::Vector::Zero()));
    alpha.setColumn(c++, KDL::Twist(KDL::Vector::Zero(), KDL::Vector(1, 0, 0)));
    alpha.setColumn(c++, KDL::Twist(KDL::Vector::Zero(), KDL::Vector(0, 1, 0)));
    alpha.setColumn(c++, KDL::Twist(KDL::Vector::Zero(), KDL::Vector(0, 0, 1)));
}

struct Controller
{
    Controller(mj_kdl::Robot &r, double gravity_z, Variant variant, bool free_z)
      : robot(r),
        variant(variant),
        free_z(free_z),
        n(r.chain.getNrOfJoints()),
        ns(r.chain.getNrOfSegments()),
        root_acc(KDL::Vector(0.0, 0.0, -gravity_z), KDL::Vector::Zero()),
        fk(r.chain),
        achd6(r.chain, root_acc, 6),
        achd5(r.chain, root_acc, 5),
        rnea(r.chain, KDL::Vector(0.0, 0.0, gravity_z)),
        q(n), qd(n), qdd(n), qd_zero(n), qdd_zero(n), ff_zero(n), ff_gravity(n),
        constraint_tau(n), tau(n),
        beta6(6), beta5(5), beta_zero(6), alpha6(6), alpha5(5), alpha_zero(6),
        f_ext(ns, KDL::Wrench::Zero()), f_ext_zero(ns, KDL::Wrench::Zero())
    {
        set_alpha(alpha6, false);
        set_alpha(alpha5, true);
        KDL::SetToZero(alpha_zero);
        KDL::SetToZero(beta_zero);
        KDL::SetToZero(ff_zero);
        KDL::SetToZero(qd_zero);
        KDL::SetToZero(qdd_zero);
        const double w = variant == Variant::Weighted ? 1.0 : 0.0;
        achd6.setDriverWeights(Eigen::VectorXd::Constant(6, w), Eigen::VectorXd::Zero(6));
        achd5.setDriverWeights(Eigen::VectorXd::Constant(5, w), Eigen::VectorXd::Zero(5));
    }

    void reset()
    {
        first = true;
        use_free_z = false;
        f_ext.back() = KDL::Wrench::Zero();
    }

    void start_press(double fz_down)
    {
        f_ext.back() = KDL::Wrench(KDL::Vector(0.0, 0.0, -fz_down), KDL::Vector::Zero());
        use_free_z = free_z;
        first = true;
    }

    unsigned active_nc() const { return use_free_z ? 5 : 6; }

    bool track(const KDL::Frame &target)
    {
        for (unsigned i = 0; i < n; ++i) {
            q(i)  = robot.jnt_pos_msr[i];
            qd(i) = robot.jnt_vel_msr[i];
        }
        KDL::Frame current;
        fk.JntToCart(q, current);
        const KDL::Twist err = KDL::diff(current, target);
        const double dt = robot.model->opt.timestep;
        const unsigned nc = active_nc();
        const unsigned n_lin = use_free_z ? 2 : 3;

        std::array<double, 6> e{};
        unsigned c = 0;
        e[c++] = err.vel.x();
        e[c++] = err.vel.y();
        if (!use_free_z) e[c++] = err.vel.z();
        e[c++] = err.rot.x();
        e[c++] = err.rot.y();
        e[c++] = err.rot.z();
        if (first) {
            err_prev = e;
            first = false;
        }
        KDL::JntArray &beta = use_free_z ? beta5 : beta6;
        for (unsigned i = 0; i < nc; ++i) {
            const double de = (e[i] - err_prev[i]) / dt;
            beta(i) = i < n_lin ? clamp_abs(kKpLin * e[i] + kKdLin * de, kBetaMax)
                                : clamp_abs(kKpRot * e[i] + kKdRot * de, kBetaMax);
        }
        err_prev = e;

        // Gravity as feed-forward: an unconstrained direction otherwise follows free fall.
        if (rnea.CartToJnt(q, qd_zero, qdd_zero, f_ext_zero, ff_gravity) < 0) return false;
        const int rc = use_free_z
          ? achd5.CartToJnt(q, qd, qdd, alpha5, beta5, f_ext, ff_gravity, constraint_tau)
          : achd6.CartToJnt(q, qd, qdd, alpha6, beta6, f_ext, ff_gravity, constraint_tau);
        if (rc < 0) return false;
        if (use_free_z) achd5.getContraintForceMagnitude(nu5); else achd6.getContraintForceMagnitude(nu6);
        if (rnea.CartToJnt(q, qd, qdd, f_ext_zero, tau) < 0) return false;
        for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = clamp_abs(tau(i), kTauMax);
        return true;
    }

    mj_kdl::Robot &robot;
    Variant variant;
    bool free_z;
    bool use_free_z = false;
    unsigned n, ns;
    KDL::Twist root_acc;
    KDL::ChainFkSolverPos_recursive fk;
    KDL::ChainHdSolver_Vereshchagin achd6, achd5;
    KDL::ChainIdSolver_RNE rnea;
    KDL::JntArray q, qd, qdd, qd_zero, qdd_zero, ff_zero, ff_gravity, constraint_tau, tau;
    KDL::JntArray beta6, beta5, beta_zero;
    KDL::Jacobian alpha6, alpha5, alpha_zero;
    KDL::Wrenches f_ext, f_ext_zero;
    Eigen::VectorXd nu6 = Eigen::VectorXd::Zero(6), nu5 = Eigen::VectorXd::Zero(5);
    std::array<double, 6> err_prev{};
    bool first = true;
};

struct Task
{
    Phase phase = Phase::Hold;
    double phase_start = 0.0;
    double last_print = -1.0;
    KDL::Frame target;
    double reaction_sum = 0.0;
    double reaction_min = 1e9;
    double reaction_max = -1e9;
    int reaction_count = 0;
};

KDL::Frame tcp_frame(Controller &ctrl)
{
    for (unsigned i = 0; i < ctrl.n; ++i) ctrl.q(i) = ctrl.robot.jnt_pos_msr[i];
    KDL::Frame out;
    ctrl.fk.JntToCart(ctrl.q, out);
    return out;
}

void start_task(Task &task, Controller &ctrl, double now)
{
    mj_kdl::update(&ctrl.robot);
    task = Task{};
    task.phase_start = now;
    task.target = tcp_frame(ctrl);
    ctrl.reset();
}

// One control tick: advances the phase machine and writes the torque command.
bool tick(Task &task, Controller &ctrl, int table_geom, double table_top_z, bool verbose)
{
    mj_kdl::Robot &robot = ctrl.robot;
    mj_kdl::update(&robot);
    const int grip = mj_name2id(robot.model, mjOBJ_ACTUATOR, kGripperActuator);
    if (grip >= 0) robot.data->ctrl[grip] = 255.0;

    const double now = robot.data->time;
    const double dt  = robot.model->opt.timestep;
    const double elapsed = now - task.phase_start;
    const double reaction = contact_normal_force(robot.model, robot.data, table_geom);

    switch (task.phase) {
    case Phase::Hold:
        if (elapsed >= kHoldTime) {
            task.phase = Phase::Descend;
            task.phase_start = now;
        }
        break;
    case Phase::Descend:
        task.target.p += KDL::Vector(0.0, 0.0, -kDescendVel * dt);
        if (reaction > kContactForce) {
            task.target = tcp_frame(ctrl);
            task.phase = Phase::Settle;
            task.phase_start = now;
        }
        break;
    case Phase::Settle:
        if (elapsed >= kHoldTime) {
            ctrl.start_press(g_press_force);
            task.phase = Phase::Press;
            task.phase_start = now;
        }
        break;
    case Phase::Press:
        if (elapsed >= kPressTime - 1.0) {
            task.reaction_sum += reaction;
            task.reaction_min = std::min(task.reaction_min, reaction);
            task.reaction_max = std::max(task.reaction_max, reaction);
            ++task.reaction_count;
        }
        break;
    }

    if (!ctrl.track(task.target)) return false;
    mj_kdl::update(&robot);

    if (verbose && now - task.last_print >= 0.5) {
        task.last_print = now;
        KDL::Frame tcp;
        mj_kdl::get_site_frame(robot.model, robot.data, "g_pinch", &tcp);
        const double cmd = task.phase == Phase::Press ? g_press_force : 0.0;
        std::cout << std::fixed << std::setprecision(3)
                  << "t=" << now << " phase=" << phase_name(task.phase)
                  << " nc=" << ctrl.active_nc()
                  << " cmd_press_N=" << cmd
                  << " table_reaction_N=" << reaction
                  << " tcp_above_table_m=" << tcp.p.z() - table_top_z
                  << " nu_lin=" << (ctrl.use_free_z ? ctrl.nu5 : ctrl.nu6).head(ctrl.use_free_z ? 2 : 3).transpose()
                  << " beta_z=" << (ctrl.use_free_z ? 0.0 : ctrl.beta6(2))
                  << " qd_norm=" << ctrl.qd.data.norm() << "\n";
    }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    bool headless = false;
    bool free_z   = false;
    Variant variant = Variant::Weighted;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--headless") headless = true;
        else if (arg == "--free-z") free_z = true;
        else if (arg == "--force" && i + 1 < argc) g_press_force = std::atof(argv[++i]);
        else if (arg == "--variant" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v == "weighted") variant = Variant::Weighted;
            else if (v == "main") variant = Variant::Main;
            else {
                std::cerr << "unknown variant '" << v << "', expected weighted or main\n";
                return 2;
            }
        } else {
            std::cerr << "usage: ex_achd_press [--variant weighted|main] [--free-z] [--force N] [--headless]\n";
            return 2;
        }
    }

    const std::string arm     = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml");
    const std::string ft      = mj_kdl_examples::asset("ft_sensor.xml");
    const std::string gripper = mj_kdl_examples::asset("robotiq_2f85/2f85.xml");
    const std::string table_p = mj_kdl_examples::asset("table.xml");

    mj_kdl::SceneObject table;
    table.name      = "table";
    table.mjcf_path = table_p;
    table.pos[2]    = kTableZ;
    table.fixed     = true;

    mj_kdl::AttachmentSpec ft_spec;
    ft_spec.mjcf_path = ft.c_str();
    ft_spec.attach_to = { mj_kdl::AttachKind::Site, "pinch_site" };

    mj_kdl::AttachmentSpec gripper_spec;
    gripper_spec.mjcf_path = gripper.c_str();
    gripper_spec.attach_to = { mj_kdl::AttachKind::Site, "wrist_ft_site" };
    gripper_spec.prefix    = "g_";

    const std::string table_top = mj_kdl::scene_object_site_name(table, "table_top");
    mj_kdl::RobotSpec robot_spec;
    robot_spec.path      = arm.c_str();
    robot_spec.attach_to = { mj_kdl::AttachKind::Site, table_top.c_str() };
    robot_spec.attachments.push_back(ft_spec);
    robot_spec.attachments.push_back(gripper_spec);

    mj_kdl::SceneSpec scene;
    scene.timestep   = 0.002;
    scene.add_floor  = true;
    scene.add_skybox = true;
    scene.objects.push_back(table);
    scene.robots.push_back(robot_spec);

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    if (!mj_kdl::build_scene(&model, &data, &scene)) return 1;

    const int table_geom = mj_name2id(model, mjOBJ_GEOM, kTableTopGeom);
    if (table_geom < 0) {
        std::cerr << "geom '" << kTableTopGeom << "' not found in the compiled scene\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    mj_kdl::ForceTorqueSensorSpec ft_sensor;
    ft_sensor.name       = "wrist_ft";
    ft_sensor.frame_site = "wrist_ft_site";
    mj_kdl::ToolFrameSpec tool;
    tool.tool_body = "g_base";
    tool.tcp_site  = "g_pinch";
    tool.ft_sensors.push_back(ft_sensor);

    mj_kdl::Robot robot;
    if (!mj_kdl::init_robot_from_mjcf(&robot, model, data, "base_link", "bracelet_link", "", &tool)) {
        mj_kdl::destroy_scene(model, data);
        return 1;
    }
    robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

    KDL::JntArray q_home(robot.n_joints);
    for (int i = 0; i < robot.n_joints; ++i) q_home(i) = kHomePose[i];
    mj_kdl::Env env;
    env.spec  = scene;
    env.model = model;
    env.data  = data;
    mj_kdl::env_add_robot(&env, &robot);
    env.on_reset = [&](mj_kdl::ResetContext *) { mj_kdl::set_joint_pos(&robot, q_home, false); };
    mj_kdl::reset(&env);

    KDL::Frame table_top_frame;
    mj_kdl::get_site_frame(model, data, table_top.c_str(), &table_top_frame);
    const double table_top_z = table_top_frame.p.z();

    Controller ctrl(robot, scene.gravity_z, variant, free_z);
    Task task;
    start_task(task, ctrl, data->time);

    std::cout << "variant=" << (variant == Variant::Weighted ? "weighted" : "main")
              << " press_constraints=" << (free_z ? 5 : 6) << (free_z ? " (linear z free)" : "")
              << " press=" << g_press_force << " N down\n";

    int status = 0;
    if (headless) {
        while (true) {
            if (!tick(task, ctrl, table_geom, table_top_z, true) || !mj_kdl::step(&robot)) {
                status = 1;
                break;
            }
            if (task.phase == Phase::Press && data->time - task.phase_start >= kPressTime) break;
            if (data->time > 60.0) {
                std::cerr << "no table contact within 60 s\n";
                status = 1;
                break;
            }
        }
        if (task.reaction_count > 0) {
            std::cout << std::fixed << std::setprecision(3)
                      << "mean_table_reaction_N=" << task.reaction_sum / task.reaction_count
                      << " min=" << task.reaction_min << " max=" << task.reaction_max
                      << " commanded_press_N=" << g_press_force << "\n";
        }
    } else {
        mj_kdl::Viewer viewer{};
        mj_kdl::set_free_camera(&viewer, 1.55, 145.0, -24.0, { 0.05, 0.0, kTableZ + 0.35 });
        const std::string title = std::string("ex_achd_press --variant ")
                                  + (variant == Variant::Weighted ? "weighted" : "main")
                                  + (free_z ? " --free-z" : "");
        if (!mj_kdl::init_window_sim(&viewer, &robot, title.c_str())) {
            mj_kdl::cleanup(&robot);
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
        const float green[4] = { 0.1f, 0.9f, 0.2f, 1.0f };
        double prev_time = data->time;
        while (mj_kdl::is_running(&viewer)) {
            if (data->time < prev_time - 1e-6) {
                mj_kdl::reset(&env);
                start_task(task, ctrl, data->time);
            }
            prev_time = data->time;
            if (!tick(task, ctrl, table_geom, table_top_z, true) || !mj_kdl::step(&robot)) break;
            if (task.phase == Phase::Press) {
                KDL::Frame tcp;
                if (mj_kdl::get_site_frame(model, data, "g_pinch", &tcp)) {
                    mj_kdl::clear_trace(&viewer);
                    mj_kdl::add_overlay_arrow(&viewer, tcp.p + KDL::Vector(0, 0, 0.25),
                                              KDL::Vector(0, 0, -1), 0.25, green);
                }
            }
            mj_kdl::pace_realtime(&robot);
        }
        mj_kdl::cleanup(&viewer);
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return status;
}
