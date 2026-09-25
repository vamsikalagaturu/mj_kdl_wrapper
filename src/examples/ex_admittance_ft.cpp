#include "admittance_ft_common.hpp"

namespace
{

class PositionController final : public admittance_ft::Controller
{
public:
    explicit PositionController(admittance_ft::SceneHandles &handles)
      : h(handles),
        fk(h.robot.chain),
        ik(h.robot.chain),
        q_des(admittance_ft::home_q(h.robot.n_joints)),
        dq(h.robot.n_joints)
    {
        ik.setLambda(0.05);
    }

    const char *name() const override { return "ex_admittance_ft"; }
    mj_kdl::CtrlMode mode() const override { return mj_kdl::CtrlMode::POSITION; }

    void reset() override { q_des = admittance_ft::home_q(h.robot.n_joints); }

    void track(const KDL::Frame &target) override
    {
        KDL::Frame current;
        fk.JntToCart(q_des, current);
        KDL::Twist dx = KDL::diff(current, target);
        const double vel_norm = dx.vel.Norm();
        const double rot_norm = dx.rot.Norm();
        if (vel_norm > 0.03) dx.vel = dx.vel * (0.03 / vel_norm);
        if (rot_norm > 0.15) dx.rot = dx.rot * (0.15 / rot_norm);

        if (ik.CartToJnt(q_des, dx, dq) >= 0) {
            for (unsigned i = 0; i < q_des.rows(); ++i) q_des(i) += dq(i);
            clamp_to_limits();
        }
        for (int i = 0; i < h.robot.n_joints; ++i) h.robot.jnt_pos_cmd[i] = q_des(i);
        mj_kdl::set_joint_pos(&h.robot, q_des);
        mj_kdl::update(&h.env);
    }

private:
    // joint_limits are +-inf for an unlimited joint, so the clamp leaves it alone.
    void clamp_to_limits()
    {
        for (int i = 0; i < h.robot.n_joints; ++i) {
            const auto [lo, hi] = h.robot.joint_limits[i];
            q_des(i)            = admittance_ft::clamp(q_des(i), lo, hi);
        }
    }

    admittance_ft::SceneHandles &h;
    KDL::ChainFkSolverPos_recursive fk;
    KDL::ChainIkSolverVel_wdls ik;
    KDL::JntArray q_des;
    KDL::JntArray dq;
};

std::unique_ptr<admittance_ft::Controller> make_controller(admittance_ft::SceneHandles &h)
{
    return std::make_unique<PositionController>(h);
}

} // namespace

int main(int argc, char **argv)
{
    return admittance_ft::run(argc, argv, make_controller);
}
