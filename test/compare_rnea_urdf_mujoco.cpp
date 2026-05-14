/* compare_rnea_urdf_mujoco.cpp
 * Builds two KDL chains for the Kinova GEN3:
 *   A) URDF  -- parsed by kdl_parser::treeFromFile (ROS-independent build)
 *   B) MuJoCo -- built by mj_kdl_wrapper::init_robot_from_mjcf
 * Runs RNEA with q=home, qdot=0, qddot=0 on both and prints per-joint
 * gravity torques side by side so any inertia discrepancy is visible. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/tree.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const double kHome[7] = {
    0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633
};

static void print_chain(const KDL::Chain &c, const char *label)
{
    std::cout << label << ": " << c.getNrOfJoints() << " joints, "
              << c.getNrOfSegments() << " segs\n";
    for (unsigned i = 0; i < c.getNrOfSegments(); ++i) {
        const auto &seg  = c.getSegment(i);
        const auto &ri   = seg.getInertia();
        const auto &I    = ri.getRotationalInertia();
        bool        fixed = seg.getJoint().getType() == KDL::Joint::None;
        std::cout << "  [" << i << "] " << seg.getName()
                  << (fixed ? " [fixed]" : " [rev]")
                  << "  m=" << std::setprecision(4) << ri.getMass()
                  << "  Ixx=" << I.data[0]
                  << " Iyy=" << I.data[4]
                  << " Izz=" << I.data[8] << "\n";
    }
    std::cout << "\n";
}

static void run_rnea(const KDL::Chain &chain, const char *label,
                     const double *q_vals, std::vector<double> &tau_out)
{
    unsigned n = chain.getNrOfJoints();
    unsigned s = chain.getNrOfSegments();

    KDL::JntArray q(n), qdot(n), qddot(n);
    for (unsigned i = 0; i < n; ++i) q(i) = q_vals[i];

    KDL::Wrenches f_ext(s, KDL::Wrench::Zero());
    KDL::JntArray torques(n);

    KDL::ChainIdSolver_RNE rnea(chain, KDL::Vector(0.0, 0.0, -9.81));
    if (rnea.CartToJnt(q, qdot, qddot, f_ext, torques) < 0) {
        std::cerr << label << ": RNEA failed\n";
        return;
    }

    tau_out.resize(n);
    std::cout << label << " (q=home, qdot=0, qddot=0):\n";
    for (unsigned i = 0; i < n; ++i) {
        tau_out[i] = torques(i);
        std::cout << "  j[" << i << "]  " << std::setw(10) << std::fixed
                  << std::setprecision(5) << torques(i) << " Nm\n";
    }

    KDL::ChainFkSolverPos_recursive fk(chain);
    KDL::Frame ee;
    fk.JntToCart(q, ee);
    std::cout << "  EE: (" << std::setprecision(4)
              << ee.p.x() << ", " << ee.p.y() << ", " << ee.p.z() << ")\n\n";
}

int main(int argc, char *argv[])
{
    fs::path root = fs::path(__FILE__).parent_path().parent_path();

    const std::string urdf_path = (argc > 1)
        ? argv[1]
        : (root / "../../motion-spec/thirdparty/kinova/GEN3_URDF_V12.urdf").string();

    const std::string arm_mjcf =
        (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf =
        (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();

    // ---- A) URDF chain ----
    KDL::Tree tree;
    if (!kdl_parser::treeFromFile(urdf_path, tree)) {
        std::cerr << "kdl_parser: failed to parse " << urdf_path << "\n";
        return 1;
    }
    KDL::Chain urdf_chain;
    if (!tree.getChain("base_link", "Bracelet_Link", urdf_chain)) {
        std::cerr << "getChain base_link->Bracelet_Link failed\n";
        return 1;
    }
    std::cout << "=== A) URDF chain ===\n";
    print_chain(urdf_chain, "URDF");

    // ---- B) MuJoCo chain ----
    mj_kdl::AttachmentSpec gripper;
    gripper.mjcf_path = grp_mjcf.c_str();
    gripper.attach_to = "bracelet_link";
    gripper.prefix    = "g_";
    gripper.pos[2]    = -0.061525;
    gripper.euler[0]  = 180.0;

    mj_kdl::RobotSpec rs;
    rs.path = arm_mjcf.c_str();
    rs.attachments.push_back(gripper);

    mj_kdl::SceneSpec sc;
    sc.robots.push_back(rs);

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    if (!mj_kdl::build_scene(&model, &data, &sc)) {
        std::cerr << "build_scene failed\n"; return 1;
    }
    mj_kdl::Robot robot;
    if (!mj_kdl::init_robot_from_mjcf(
          &robot, model, data, "base_link", "bracelet_link", "", nullptr)) {
        std::cerr << "init_robot_from_mjcf failed\n";
        mj_kdl::destroy_scene(model, data); return 1;
    }

    std::cout << "=== B) MuJoCo chain (arm, no gripper tool) ===\n";
    print_chain(robot.chain, "MuJoCo");

    // ---- compare ----
    std::cout << "=== RNEA gravity torques ===\n\n";
    std::vector<double> tau_urdf, tau_mj;
    run_rnea(urdf_chain, "URDF  ", kHome, tau_urdf);
    run_rnea(robot.chain, "MuJoCo", kHome, tau_mj);

    unsigned n = std::min(tau_urdf.size(), tau_mj.size());
    std::cout << "=== Difference (MuJoCo - URDF) ===\n";
    for (unsigned i = 0; i < n; ++i) {
        double d = tau_mj[i] - tau_urdf[i];
        std::cout << "  j[" << i << "]  " << std::setw(9) << std::fixed
                  << std::setprecision(5) << d << " Nm"
                  << (std::abs(d) > 0.05 ? "  <-- MISMATCH" : "") << "\n";
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
