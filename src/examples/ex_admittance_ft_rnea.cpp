#include "admittance_ft_common.hpp"

namespace
{

static constexpr double kKpLin = 11520.0;
static constexpr double kKdLin = 514.3;
static constexpr double kKpRot = 7200.0;
static constexpr double kKdRot = 600.1;
static constexpr double kBetaLinMax = 7200.0;
static constexpr double kBetaRotMax = 5040.0;
static constexpr double kTauMax = 141.6;

class RneaTaskController final : public admittance_ft::Controller
{
public:
    explicit RneaTaskController(admittance_ft::SceneHandles &handles)
      : h(handles),
        fk(h.robot.chain),
        jac_solver(h.robot.chain),
        ik_acc(h.robot.chain),
        rnea(h.robot.chain, KDL::Vector(0.0, 0.0, h.scene.gravity_z)),
        q(h.robot.n_joints),
        qd(h.robot.n_joints),
        qdd(h.robot.n_joints),
        tau(h.robot.n_joints),
        jac(h.robot.n_joints),
        f_ext(h.robot.chain.getNrOfSegments(), KDL::Wrench::Zero())
    {
        ik_acc.setLambda(0.10);
    }

    const char *name() const override { return "ex_admittance_ft_rnea"; }
    mj_kdl::CtrlMode mode() const override { return mj_kdl::CtrlMode::TORQUE; }
    void reset() override {}

    void track(const KDL::Frame &target) override
    {
        fill_state();
        KDL::Frame current;
        fk.JntToCart(q, current);
        const KDL::Twist err = KDL::diff(current, target);

        jac_solver.JntToJac(q, jac);
        const KDL::Twist tcp_vel = jacobian_twist();

        const KDL::Twist beta(
          KDL::Vector(
            admittance_ft::clamp(kKpLin * err.vel.x() - kKdLin * tcp_vel.vel.x(), -kBetaLinMax, kBetaLinMax),
            admittance_ft::clamp(kKpLin * err.vel.y() - kKdLin * tcp_vel.vel.y(), -kBetaLinMax, kBetaLinMax),
            admittance_ft::clamp(kKpLin * err.vel.z() - kKdLin * tcp_vel.vel.z(), -kBetaLinMax, kBetaLinMax)
          ),
          KDL::Vector(
            admittance_ft::clamp(kKpRot * err.rot.x() - kKdRot * tcp_vel.rot.x(), -kBetaRotMax, kBetaRotMax),
            admittance_ft::clamp(kKpRot * err.rot.y() - kKdRot * tcp_vel.rot.y(), -kBetaRotMax, kBetaRotMax),
            admittance_ft::clamp(kKpRot * err.rot.z() - kKdRot * tcp_vel.rot.z(), -kBetaRotMax, kBetaRotMax)
          )
        );

        if (ik_acc.CartToJnt(q, beta, qdd) < 0) return;
        if (rnea.CartToJnt(q, qd, qdd, f_ext, tau) < 0) return;
        for (int i = 0; i < h.robot.n_joints; ++i) {
            h.robot.jnt_trq_cmd[i] = admittance_ft::clamp(tau(i), -kTauMax, kTauMax);
        }
    }

private:
    void fill_state()
    {
        for (int i = 0; i < h.robot.n_joints; ++i) {
            q(i) = h.robot.jnt_pos_msr[i];
            qd(i) = h.robot.jnt_vel_msr[i];
        }
    }

    KDL::Twist jacobian_twist() const
    {
        KDL::Twist out = KDL::Twist::Zero();
        for (unsigned j = 0; j < q.rows(); ++j) {
            out.vel.x(out.vel.x() + jac(0, j) * qd(j));
            out.vel.y(out.vel.y() + jac(1, j) * qd(j));
            out.vel.z(out.vel.z() + jac(2, j) * qd(j));
            out.rot.x(out.rot.x() + jac(3, j) * qd(j));
            out.rot.y(out.rot.y() + jac(4, j) * qd(j));
            out.rot.z(out.rot.z() + jac(5, j) * qd(j));
        }
        return out;
    }

    admittance_ft::SceneHandles &h;
    KDL::ChainFkSolverPos_recursive fk;
    KDL::ChainJntToJacSolver jac_solver;
    KDL::ChainIkSolverVel_wdls ik_acc;
    KDL::ChainIdSolver_RNE rnea;
    KDL::JntArray q;
    KDL::JntArray qd;
    KDL::JntArray qdd;
    KDL::JntArray tau;
    KDL::Jacobian jac;
    KDL::Wrenches f_ext;
};

std::unique_ptr<admittance_ft::Controller> make_controller(admittance_ft::SceneHandles &h)
{
    return std::make_unique<RneaTaskController>(h);
}

} // namespace

int main(int argc, char **argv)
{
    return admittance_ft::run(argc, argv, make_controller);
}
