#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <mujoco/mujoco.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using mj_kdl::AttachKind;
using mj_kdl::AttachTarget;
using mj_kdl::CameraSpec;
using mj_kdl::Condim;
using mj_kdl::CtrlMode;
using mj_kdl::RobotSpec;
using mj_kdl::SceneObject;
using mj_kdl::SceneSpec;
using mj_kdl::Shape;

struct PyAttachTarget
{
    AttachKind  kind = AttachKind::World;
    std::string name;
};

struct PyAttachmentSpec
{
    std::string                                      mjcf_path;
    PyAttachTarget                                   attach_to;
    std::string                                      prefix;
    std::array<double, 3>                            pos   = { 0.0, 0.0, 0.0 };
    std::array<double, 3>                            euler = { 0.0, 0.0, 0.0 };
    std::vector<std::pair<std::string, std::string>> contact_exclusions;
};

struct PyRobotSpec
{
    std::string                   path;
    std::string                   prefix;
    PyAttachTarget                attach_to;
    std::array<double, 3>         pos   = { 0.0, 0.0, 0.0 };
    std::array<double, 3>         euler = { 0.0, 0.0, 0.0 };
    std::vector<PyAttachmentSpec> attachments;
};

struct PySceneObject
{
    std::string                          name;
    std::string                          mjcf_path;
    PyAttachTarget                       attach_to;
    Shape                                shape = Shape::Unspecified;
    std::optional<std::array<double, 3>> size;
    std::array<double, 3>                pos = { 0.0, 0.0, 0.0 };
    std::optional<std::array<float, 4>>  rgba;
    bool                                 fixed = false;
    std::optional<double>                mass;
    Condim                               condim = Condim::Tangential;
    std::optional<std::array<double, 3>> friction;
};

struct PyCameraSpec
{
    std::string                          name;
    std::optional<std::array<double, 3>> pos;
    std::array<double, 3>                euler = { 0.0, 0.0, 0.0 };
    std::optional<double>                fovy;
};

struct PySceneSpec
{
    std::vector<PyRobotSpec>   robots;
    std::optional<double>      timestep;
    double                     gravity_z = -9.81;
    std::optional<bool>        add_floor;
    std::optional<bool>        add_skybox;
    std::vector<PySceneObject> objects;
    std::vector<PyCameraSpec>  cameras;
};

struct PyToolFrameSpec
{
    std::string tool_body;
    std::string tcp_site;
};

struct PyScene;
struct PyEnv;
struct PyRobot;

AttachTarget to_cpp(const PyAttachTarget &src)
{
    AttachTarget out;
    out.kind = src.kind;
    out.name = (src.kind == AttachKind::World || src.name.empty()) ? nullptr : src.name.c_str();
    return out;
}

mj_kdl::AttachmentSpec to_cpp(const PyAttachmentSpec &src)
{
    mj_kdl::AttachmentSpec out;
    out.mjcf_path = src.mjcf_path.empty() ? nullptr : src.mjcf_path.c_str();
    out.attach_to = to_cpp(src.attach_to);
    out.prefix    = src.prefix.c_str();
    std::copy(src.pos.begin(), src.pos.end(), out.pos);
    std::copy(src.euler.begin(), src.euler.end(), out.euler);
    out.contact_exclusions = src.contact_exclusions;
    return out;
}

RobotSpec to_cpp(const PyRobotSpec &src)
{
    RobotSpec out;
    out.path      = src.path.empty() ? nullptr : src.path.c_str();
    out.prefix    = src.prefix.c_str();
    out.attach_to = to_cpp(src.attach_to);
    std::copy(src.pos.begin(), src.pos.end(), out.pos);
    std::copy(src.euler.begin(), src.euler.end(), out.euler);
    out.attachments.reserve(src.attachments.size());
    for (const auto &item : src.attachments) out.attachments.push_back(to_cpp(item));
    return out;
}

SceneObject to_cpp(const PySceneObject &src)
{
    SceneObject out;
    out.name      = src.name;
    out.mjcf_path = src.mjcf_path;
    out.attach_to = to_cpp(src.attach_to);
    out.shape     = src.shape;
    std::copy(src.pos.begin(), src.pos.end(), out.pos);
    out.fixed  = src.fixed;
    out.condim = src.condim;
    if (src.mjcf_path.empty()) {
        if (!src.size)
            throw std::runtime_error("SceneObject.size must be set for primitive objects");
        if (!src.rgba)
            throw std::runtime_error("SceneObject.rgba must be set for primitive objects");
        if (!src.mass)
            throw std::runtime_error("SceneObject.mass must be set for primitive objects");
        if (!src.friction) {
            throw std::runtime_error("SceneObject.friction must be set for primitive objects");
        }
        std::copy(src.size->begin(), src.size->end(), out.size);
        std::copy(src.rgba->begin(), src.rgba->end(), out.rgba);
        out.mass = *src.mass;
        std::copy(src.friction->begin(), src.friction->end(), out.friction);
    }
    return out;
}

CameraSpec to_cpp(const PyCameraSpec &src)
{
    CameraSpec out;
    out.name = src.name;
    if (!src.pos) throw std::runtime_error("CameraSpec.pos must be set");
    if (!src.fovy) throw std::runtime_error("CameraSpec.fovy must be set");
    std::copy(src.pos->begin(), src.pos->end(), out.pos);
    std::copy(src.euler.begin(), src.euler.end(), out.euler);
    out.fovy = *src.fovy;
    return out;
}

SceneSpec to_cpp(const PySceneSpec &src)
{
    SceneSpec out;
    if (!src.timestep) throw std::runtime_error("SceneSpec.timestep must be set");
    if (!src.add_floor) throw std::runtime_error("SceneSpec.add_floor must be set");
    if (!src.add_skybox) throw std::runtime_error("SceneSpec.add_skybox must be set");
    out.timestep   = *src.timestep;
    out.gravity_z  = src.gravity_z;
    out.add_floor  = *src.add_floor;
    out.add_skybox = *src.add_skybox;
    out.robots.reserve(src.robots.size());
    for (const auto &item : src.robots) out.robots.push_back(to_cpp(item));
    out.objects.reserve(src.objects.size());
    for (const auto &item : src.objects) out.objects.push_back(to_cpp(item));
    out.cameras.reserve(src.cameras.size());
    for (const auto &item : src.cameras) out.cameras.push_back(to_cpp(item));
    return out;
}

KDL::JntArray to_jnt_array(const std::vector<double> &values, int expected)
{
    if (static_cast<int>(values.size()) != expected) {
        throw std::invalid_argument(
          "expected " + std::to_string(expected) + " joint values, got "
          + std::to_string(values.size())
        );
    }
    KDL::JntArray out(expected);
    for (int i = 0; i < expected; ++i) out(i) = values[static_cast<size_t>(i)];
    return out;
}

KDL::JntArray to_jnt_array(const py::object &values, int expected)
{
    if (py::hasattr(values, "rows")) {
        int rows = values.attr("rows")().cast<int>();
        if (rows != expected) {
            throw std::invalid_argument(
              "expected " + std::to_string(expected) + " joint values, got "
              + std::to_string(rows)
            );
        }
        KDL::JntArray out(expected);
        for (int i = 0; i < expected; ++i) out(i) = values[py::int_(i)].cast<double>();
        return out;
    }
    return to_jnt_array(values.cast<std::vector<double>>(), expected);
}

py::object kdl_frame_to_py(const KDL::Frame &frame)
{
    py::module_ kdl = py::module_::import("PyKDL");
    double      x = 0.0, y = 0.0, z = 0.0, w = 1.0;
    frame.M.GetQuaternion(x, y, z, w);
    py::object rotation = kdl.attr("Rotation").attr("Quaternion")(x, y, z, w);
    py::object vector   = kdl.attr("Vector")(frame.p.x(), frame.p.y(), frame.p.z());
    return kdl.attr("Frame")(rotation, vector);
}

py::object kdl_chain_to_py(const KDL::Chain &chain)
{
    py::module_::import("PyKDL");
    try {
        return py::cast(chain);
    } catch (const py::cast_error &err) {
        throw std::runtime_error(
          std::string("PyKDL did not register a compatible KDL::Chain caster: ") + err.what()
        );
    }
}

struct PyScene : std::enable_shared_from_this<PyScene>
{
    PySceneSpec spec;
    mjModel    *model = nullptr;
    mjData     *data  = nullptr;
    std::vector<std::weak_ptr<PyRobot>> robots;

    ~PyScene() { close(); }

    void register_robot(const std::shared_ptr<PyRobot> &robot);
    void reinit_robots();
    void invalidate_robots();

    static std::shared_ptr<PyScene> build(const PySceneSpec &spec)
    {
        auto scene         = std::shared_ptr<PyScene>(new PyScene());
        scene->spec        = spec;
        SceneSpec cpp_spec = to_cpp(scene->spec);
        if (!mj_kdl::build_scene(&scene->model, &scene->data, &cpp_spec)) {
            throw std::runtime_error("build_scene failed");
        }
        return scene;
    }

    void close()
    {
        invalidate_robots();
        mj_kdl::destroy_scene(model, data);
        model = nullptr;
        data  = nullptr;
    }

    void save_xml(const std::string &path) const
    {
        if (!model) throw std::runtime_error("scene is closed");
        if (!mj_kdl::save_model_xml(model, path.c_str())) {
            throw std::runtime_error("save_model_xml failed");
        }
    }

    void save_binary(const std::string &path) const
    {
        if (!model) throw std::runtime_error("scene is closed");
        mj_saveModel(model, path.c_str(), nullptr, 0);
    }

    double time() const
    {
        if (!data) throw std::runtime_error("scene is closed");
        return data->time;
    }

    double timestep() const
    {
        if (!model) throw std::runtime_error("scene is closed");
        return model->opt.timestep;
    }

    void step()
    {
        if (!model || !data) throw std::runtime_error("scene is closed");
        mj_step(model, data);
    }

    void step_n(int n)
    {
        if (n < 0) throw std::invalid_argument("n must be non-negative");
        for (int i = 0; i < n; ++i) step();
    }

    std::vector<std::string> camera_names() const
    {
        if (!model) throw std::runtime_error("scene is closed");
        return mj_kdl::get_camera_names(model);
    }

    py::object body_frame(const std::string &name)
    {
        if (!model || !data) throw std::runtime_error("scene is closed");
        KDL::Frame frame;
        if (!mj_kdl::get_body_frame(model, data, name.c_str(), &frame)) {
            throw std::runtime_error("body not found");
        }
        return kdl_frame_to_py(frame);
    }

    py::object site_frame(const std::string &name)
    {
        if (!model || !data) throw std::runtime_error("scene is closed");
        KDL::Frame frame;
        if (!mj_kdl::get_site_frame(model, data, name.c_str(), &frame)) {
            throw std::runtime_error("site not found");
        }
        return kdl_frame_to_py(frame);
    }

    void set_body_pose(
      const std::string           &name,
      const std::array<double, 3> &pos,
      const std::array<double, 4> *quat
    )
    {
        if (!model || !data) throw std::runtime_error("scene is closed");
        mj_kdl::set_body_pose(model, data, name.c_str(), pos.data(), quat ? quat->data() : nullptr);
    }

    void set_actuator_ctrl(const std::string &name, double value)
    {
        if (!model || !data) throw std::runtime_error("scene is closed");
        int id = mj_name2id(model, mjOBJ_ACTUATOR, name.c_str());
        if (id < 0) throw std::runtime_error("actuator not found: " + name);
        data->ctrl[id] = value;
    }

    double actuator_ctrl(const std::string &name) const
    {
        if (!model || !data) throw std::runtime_error("scene is closed");
        int id = mj_name2id(model, mjOBJ_ACTUATOR, name.c_str());
        if (id < 0) throw std::runtime_error("actuator not found: " + name);
        return data->ctrl[id];
    }

    bool has_actuator(const std::string &name) const
    {
        if (!model) throw std::runtime_error("scene is closed");
        return mj_name2id(model, mjOBJ_ACTUATOR, name.c_str()) >= 0;
    }

    void add_object(const PySceneObject &object)
    {
        PySceneSpec next_spec = spec;
        next_spec.objects.push_back(object);

        mjModel  *next_model = nullptr;
        mjData   *next_data  = nullptr;
        SceneSpec cpp_spec   = to_cpp(next_spec);
        if (!mj_kdl::build_scene(&next_model, &next_data, &cpp_spec)) {
            throw std::runtime_error("add_object rebuild failed");
        }

        mj_kdl::destroy_scene(model, data);
        spec  = std::move(next_spec);
        model = next_model;
        data  = next_data;
        reinit_robots();
    }

    void remove_object(const std::string &name)
    {
        PySceneSpec next_spec = spec;
        auto        it =
          std::find_if(next_spec.objects.begin(), next_spec.objects.end(), [&](const auto &obj) {
              return obj.name == name;
          });
        if (it == next_spec.objects.end()) throw std::runtime_error("object not found");
        next_spec.objects.erase(it);

        mjModel  *next_model = nullptr;
        mjData   *next_data  = nullptr;
        SceneSpec cpp_spec   = to_cpp(next_spec);
        if (!mj_kdl::build_scene(&next_model, &next_data, &cpp_spec)) {
            throw std::runtime_error("remove_object rebuild failed");
        }

        mj_kdl::destroy_scene(model, data);
        spec  = std::move(next_spec);
        model = next_model;
        data  = next_data;
        reinit_robots();
    }

};

struct PyRobot
{
    mj_kdl::Robot            robot;
    std::shared_ptr<PyScene> scene_owner;
    std::shared_ptr<PyEnv>   env_owner;
    std::string              base_body;
    std::string              tip_body;
    std::string              prefix;
    std::optional<PyToolFrameSpec> tool_spec;
    bool                     active = false;

    ~PyRobot();

    static std::shared_ptr<PyRobot> from_scene(
      const std::shared_ptr<PyScene> &scene,
      const std::string              &base_body,
      const std::string              &tip_body,
      const std::string              &prefix,
      const PyToolFrameSpec          *tool
    )
    {
        if (!scene || !scene->model || !scene->data) throw std::runtime_error("scene is closed");
        auto out         = std::shared_ptr<PyRobot>(new PyRobot());
        out->scene_owner = scene;
        out->init(scene->model, scene->data, base_body, tip_body, prefix, tool);
        scene->register_robot(out);
        return out;
    }

    void init(
      mjModel               *model,
      mjData                *data,
      const std::string     &base_body_arg,
      const std::string     &tip_body_arg,
      const std::string     &prefix_arg,
      const PyToolFrameSpec *tool
    )
    {
        active    = false;
        base_body = base_body_arg;
        tip_body  = tip_body_arg;
        prefix    = prefix_arg;
        std::optional<PyToolFrameSpec> next_tool;
        if (tool) next_tool = *tool;
        tool_spec = std::move(next_tool);
        const PyToolFrameSpec *stored_tool = tool_spec ? &*tool_spec : nullptr;

        mj_kdl::ToolFrameSpec        cpp_tool;
        const mj_kdl::ToolFrameSpec *tool_ptr = nullptr;
        if (stored_tool) {
            cpp_tool.tool_body =
              stored_tool->tool_body.empty() ? nullptr : stored_tool->tool_body.c_str();
            cpp_tool.tcp_site =
              stored_tool->tcp_site.empty() ? nullptr : stored_tool->tcp_site.c_str();
            tool_ptr           = &cpp_tool;
        }
        if (!mj_kdl::init_robot_from_mjcf(
              &robot,
              model,
              data,
              base_body_arg.c_str(),
              tip_body_arg.c_str(),
              prefix_arg.c_str(),
              tool_ptr
            )) {
            throw std::runtime_error("init_robot_from_mjcf failed");
        }
        active = true;
    }

    void ensure_active() const
    {
        if (!active || !robot.model || !robot.data || robot.n_joints <= 0) {
            throw std::runtime_error("robot is closed");
        }
    }

    void invalidate()
    {
        mj_kdl::cleanup(&robot);
        active = false;
    }

    void reinit(mjModel *model, mjData *data)
    {
        if (base_body.empty() || tip_body.empty()) {
            throw std::runtime_error("robot init parameters are missing");
        }
        ensure_active();
        CtrlMode            ctrl_mode = robot.ctrl_mode;
        bool                paused    = robot.paused;
        std::vector<double> pos_cmd   = robot.jnt_pos_cmd;
        std::vector<double> trq_cmd   = robot.jnt_trq_cmd;

        invalidate();
        init(model, data, base_body, tip_body, prefix, tool_spec ? &*tool_spec : nullptr);
        robot.ctrl_mode = ctrl_mode;
        robot.paused    = paused;
        if (static_cast<int>(pos_cmd.size()) == robot.n_joints) robot.jnt_pos_cmd = pos_cmd;
        if (static_cast<int>(trq_cmd.size()) == robot.n_joints) robot.jnt_trq_cmd = trq_cmd;
    }

    void update()
    {
        ensure_active();
        mj_kdl::update(&robot);
    }

    bool step()
    {
        ensure_active();
        return mj_kdl::step(&robot);
    }

    bool step_n(int n)
    {
        ensure_active();
        if (n < 0) throw std::invalid_argument("n must be non-negative");
        return mj_kdl::step_n(&robot, n);
    }

    void set_joint_pos(const KDL::JntArray &joints, bool call_forward)
    {
        ensure_active();
        if (static_cast<int>(joints.rows()) != robot.n_joints) {
            throw std::invalid_argument(
              "expected " + std::to_string(robot.n_joints) + " joint values, got "
              + std::to_string(joints.rows())
            );
        }
        mj_kdl::set_joint_pos(&robot, joints, call_forward);
    }

    void set_joint_pos(const std::vector<double> &q, bool call_forward)
    {
        set_joint_pos(to_jnt_array(q, robot.n_joints), call_forward);
    }

    std::vector<std::pair<double, double>> joint_limits() const
    {
        ensure_active();
        return robot.joint_limits;
    }

    std::vector<double> gravity_torques(double gravity_z) const
    {
        ensure_active();
        KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0.0, 0.0, gravity_z));
        KDL::JntArray      q(robot.n_joints);
        KDL::JntArray      g(robot.n_joints);
        for (int i = 0; i < robot.n_joints; ++i) q(i) = robot.jnt_pos_msr[static_cast<size_t>(i)];
        int rc = dyn.JntToGravity(q, g);
        if (rc < 0) throw std::runtime_error("KDL::ChainDynParam::JntToGravity failed");
        std::vector<double> out(static_cast<size_t>(robot.n_joints));
        for (int i = 0; i < robot.n_joints; ++i) out[static_cast<size_t>(i)] = g(i);
        return out;
    }

    py::object fk_frame(const KDL::JntArray *q_values) const
    {
        ensure_active();
        KDL::JntArray q(robot.n_joints);
        if (q_values) {
            if (static_cast<int>(q_values->rows()) != robot.n_joints) {
                throw std::invalid_argument(
                  "expected " + std::to_string(robot.n_joints) + " joint values, got "
                  + std::to_string(q_values->rows())
                );
            }
            q = *q_values;
        } else {
            for (int i = 0; i < robot.n_joints; ++i) {
                q(i) = robot.jnt_pos_msr[static_cast<size_t>(i)];
            }
        }
        KDL::ChainFkSolverPos_recursive fk(robot.chain);
        KDL::Frame                      frame;
        int                             rc = fk.JntToCart(q, frame);
        if (rc < 0) throw std::runtime_error("KDL::ChainFkSolverPos_recursive::JntToCart failed");
        return kdl_frame_to_py(frame);
    }

    py::object kdl_chain() const
    {
        ensure_active();
        return kdl_chain_to_py(robot.chain);
    }

    void set_port(std::vector<double> mj_kdl::Robot::*port, const std::vector<double> &values)
    {
        ensure_active();
        if (static_cast<int>(values.size()) != robot.n_joints) {
            throw std::invalid_argument(
              "expected " + std::to_string(robot.n_joints) + " joint values, got "
              + std::to_string(values.size())
            );
        }
        robot.*port = values;
    }
};

void PyScene::register_robot(const std::shared_ptr<PyRobot> &robot)
{
    robots.push_back(robot);
}

void PyScene::reinit_robots()
{
    auto it = robots.begin();
    while (it != robots.end()) {
        if (auto robot = it->lock()) {
            robot->reinit(model, data);
            ++it;
        } else {
            it = robots.erase(it);
        }
    }
}

void PyScene::invalidate_robots()
{
    auto it = robots.begin();
    while (it != robots.end()) {
        if (auto robot = it->lock()) {
            robot->invalidate();
            ++it;
        } else {
            it = robots.erase(it);
        }
    }
}

struct PySimulateViewer
{
    mj_kdl::Viewer           viewer;
    std::shared_ptr<PyRobot> robot_owner;
    bool                     active = false;

    ~PySimulateViewer() { close(); }

    static std::shared_ptr<PySimulateViewer>
      open(const std::shared_ptr<PyRobot> &robot, const std::string &title)
    {
        if (!robot) throw std::runtime_error("robot is null");
        if (!robot->robot.model || !robot->robot.data) throw std::runtime_error("robot is closed");

        auto out         = std::shared_ptr<PySimulateViewer>(new PySimulateViewer());
        out->robot_owner = robot;
        if (!mj_kdl::init_window_sim(&out->viewer, &out->robot_owner->robot, title.c_str())) {
            throw std::runtime_error("init_window_sim failed");
        }
        out->active = true;
        return out;
    }

    void close()
    {
        if (!active) return;
        mj_kdl::cleanup(&viewer);
        active = false;
        robot_owner.reset();
    }

    bool is_running() const
    {
        if (!active) return false;
        return mj_kdl::is_running(&viewer);
    }

    bool step()
    {
        if (!active || !robot_owner) throw std::runtime_error("viewer is closed");
        return mj_kdl::step(&robot_owner->robot);
    }

    bool step_n(int n)
    {
        if (n < 0) throw std::invalid_argument("n must be non-negative");
        for (int i = 0; i < n; ++i) {
            if (!step()) return false;
        }
        return true;
    }

    void clear_trace()
    {
        if (!active) throw std::runtime_error("viewer is closed");
        mj_kdl::clear_trace(&viewer);
    }

    void add_trace_segment(
      const std::array<double, 3> &a,
      const std::array<double, 3> &b,
      const std::array<float, 4>  *rgba
    )
    {
        if (!active) throw std::runtime_error("viewer is closed");
        KDL::Vector ka(a[0], a[1], a[2]);
        KDL::Vector kb(b[0], b[1], b[2]);
        mj_kdl::add_trace_segment(&viewer, ka, kb, rgba ? rgba->data() : nullptr);
    }

    bool use_camera(const std::string &name)
    {
        if (!active || !robot_owner) throw std::runtime_error("viewer is closed");
        return mj_kdl::use_camera(
          &viewer, robot_owner->robot.model, name.empty() ? nullptr : name.c_str()
        );
    }
};

struct PyVideoRecorder
{
    mj_kdl::VideoRecorder    recorder;
    std::shared_ptr<PyScene> scene_owner;
    bool                     active = false;

    ~PyVideoRecorder() { close(); }

    static std::shared_ptr<PyVideoRecorder> open(
      const std::shared_ptr<PyScene> &scene,
      const std::string              &out_path,
      int                             width,
      int                             height,
      int                             fps
    )
    {
        if (!scene || !scene->model || !scene->data) throw std::runtime_error("scene is closed");
        auto out         = std::shared_ptr<PyVideoRecorder>(new PyVideoRecorder());
        out->scene_owner = scene;
        if (!mj_kdl::init_video_recorder(
              &out->recorder, scene->model, out_path.c_str(), width, height, fps
            )) {
            throw std::runtime_error("init_video_recorder failed");
        }
        out->active = true;
        return out;
    }

    static std::shared_ptr<PyVideoRecorder> open_preset(
      const std::shared_ptr<PyScene> &scene,
      const std::string              &out_path,
      mj_kdl::VideoResolution         resolution,
      int                             fps
    )
    {
        if (!scene || !scene->model || !scene->data) throw std::runtime_error("scene is closed");
        auto out         = std::shared_ptr<PyVideoRecorder>(new PyVideoRecorder());
        out->scene_owner = scene;
        if (!mj_kdl::init_video_recorder(
              &out->recorder, scene->model, out_path.c_str(), resolution, fps
            )) {
            throw std::runtime_error("init_video_recorder failed");
        }
        out->active = true;
        return out;
    }

    bool record_frame()
    {
        if (!active || !scene_owner || !scene_owner->model || !scene_owner->data) {
            throw std::runtime_error("recorder is closed");
        }
        return mj_kdl::record_frame(&recorder, scene_owner->model, scene_owner->data);
    }

    bool use_camera(const std::string &name)
    {
        if (!active || !scene_owner || !scene_owner->model) {
            throw std::runtime_error("recorder is closed");
        }
        return mj_kdl::use_camera(
          &recorder, scene_owner->model, name.empty() ? nullptr : name.c_str()
        );
    }

    void close()
    {
        if (!active) return;
        mj_kdl::cleanup(&recorder);
        active = false;
        scene_owner.reset();
    }
};

struct PyEnv : std::enable_shared_from_this<PyEnv>
{
    PySceneSpec spec;
    mj_kdl::Env env;
    std::vector<std::weak_ptr<PyRobot>> robots;

    ~PyEnv() { close(); }

    static std::shared_ptr<PyEnv> build(const PySceneSpec &spec)
    {
        auto out           = std::shared_ptr<PyEnv>(new PyEnv());
        out->spec          = spec;
        SceneSpec cpp_spec = to_cpp(out->spec);
        if (!mj_kdl::init_env(&out->env, &cpp_spec)) throw std::runtime_error("init_env failed");
        return out;
    }

    void close()
    {
        invalidate_robots();
        mj_kdl::cleanup(&env);
    }

    void reinit_robots()
    {
        env.robots.clear();
        auto it = robots.begin();
        while (it != robots.end()) {
            if (auto robot = it->lock()) {
                robot->reinit(env.model, env.data);
                mj_kdl::env_add_robot(&env, &robot->robot);
                ++it;
            } else {
                it = robots.erase(it);
            }
        }
    }

    void invalidate_robots()
    {
        auto it = robots.begin();
        while (it != robots.end()) {
            if (auto robot = it->lock()) {
                robot->invalidate();
                ++it;
            } else {
                it = robots.erase(it);
            }
        }
        env.robots.clear();
    }

    void rebuild(PySceneSpec next_spec, const char *error_message)
    {
        PySceneSpec old_spec = spec;
        spec                 = std::move(next_spec);

        mj_kdl::Env next_env;
        SceneSpec   cpp_spec = to_cpp(spec);
        if (!mj_kdl::init_env(&next_env, &cpp_spec)) {
            spec = std::move(old_spec);
            SceneSpec old_cpp_spec = to_cpp(spec);
            env.spec               = old_cpp_spec;
            throw std::runtime_error(error_message);
        }

        mj_kdl::cleanup(&env);
        env = next_env;
        reinit_robots();
    }

    std::shared_ptr<PyRobot> create_robot(
      const std::string     &base_body,
      const std::string     &tip_body,
      const std::string     &prefix,
      const PyToolFrameSpec *tool
    )
    {
        if (!env.model || !env.data) throw std::runtime_error("env is closed");
        auto out       = std::shared_ptr<PyRobot>(new PyRobot());
        out->env_owner = shared_from_this();
        out->init(env.model, env.data, base_body, tip_body, prefix, tool);
        mj_kdl::env_add_robot(&env, &out->robot);
        robots.push_back(out);
        return out;
    }

    mj_kdl::ResetInfo reset(const mj_kdl::ResetOptions *options)
    {
        if (!env.model || !env.data) throw std::runtime_error("env is closed");
        return mj_kdl::reset(&env, options);
    }

    void add_object(const PySceneObject &object)
    {
        if (!env.model || !env.data) throw std::runtime_error("env is closed");
        PySceneSpec next_spec = spec;
        next_spec.objects.push_back(object);
        rebuild(std::move(next_spec), "scene_add_object failed");
    }

    void remove_object(const std::string &name)
    {
        if (!env.model || !env.data) throw std::runtime_error("env is closed");
        PySceneSpec next_spec = spec;
        auto it = std::find_if(next_spec.objects.begin(), next_spec.objects.end(), [&](const auto &obj) {
            return obj.name == name;
        });
        if (it == next_spec.objects.end()) throw std::runtime_error("object not found");
        next_spec.objects.erase(it);
        rebuild(std::move(next_spec), "scene_remove_object failed");
    }

    std::vector<std::string> camera_names() const
    {
        if (!env.model) throw std::runtime_error("env is closed");
        return mj_kdl::get_camera_names(env.model);
    }
};

PyRobot::~PyRobot()
{
    if (env_owner) {
        auto &robots = env_owner->env.robots;
        robots.erase(std::remove(robots.begin(), robots.end(), &robot), robots.end());
    }
    invalidate();
}

} // namespace

PYBIND11_MODULE(_mj_kdl_wrapper, m)
{
    m.doc()                      = "Python bindings for mj_kdl_wrapper";
    m.attr("__version__")        = "0.1.0";
    m.attr("__mujoco_version__") = mj_versionString();

    py::enum_<mj_kdl::LogLevel>(m, "LogLevel")
      .value("NONE", mj_kdl::LogLevel::NONE)
      .value("INFO", mj_kdl::LogLevel::INFO)
      .value("WARN", mj_kdl::LogLevel::WARN)
      .value("ERROR", mj_kdl::LogLevel::ERROR);

    py::enum_<AttachKind>(m, "AttachKind")
      .value("World", AttachKind::World)
      .value("Body", AttachKind::Body)
      .value("Site", AttachKind::Site)
      .value("Frame", AttachKind::Frame);

    py::enum_<Shape>(m, "Shape")
      .value("Unspecified", Shape::Unspecified)
      .value("BOX", Shape::BOX)
      .value("SPHERE", Shape::SPHERE)
      .value("CYLINDER", Shape::CYLINDER);

    py::enum_<Condim>(m, "Condim")
      .value("Tangential", Condim::Tangential)
      .value("Torsional", Condim::Torsional)
      .value("Rolling", Condim::Rolling);

    py::enum_<CtrlMode>(m, "CtrlMode")
      .value("POSITION", CtrlMode::POSITION)
      .value("TORQUE", CtrlMode::TORQUE);

    py::enum_<mj_kdl::VideoResolution>(m, "VideoResolution")
      .value("R360p", mj_kdl::VideoResolution::R360p)
      .value("R480p", mj_kdl::VideoResolution::R480p)
      .value("R720p", mj_kdl::VideoResolution::R720p)
      .value("R1080p", mj_kdl::VideoResolution::R1080p)
      .value("R2K", mj_kdl::VideoResolution::R2K)
      .value("R4K", mj_kdl::VideoResolution::R4K);

    py::class_<PyAttachTarget>(
      m,
      "AttachTarget",
      "Placement target in an accumulated scene spec. World is the default; other kinds use name."
    )
      .def(py::init<>())
      .def(
        py::init([](AttachKind kind, const std::string &name) {
            PyAttachTarget out;
            out.kind = kind;
            out.name = name;
            return out;
        }),
        py::arg("kind") = AttachKind::World,
        py::arg("name") = ""
      )
      .def_readwrite("kind", &PyAttachTarget::kind, "Target kind: World, Body, Site, or Frame.")
      .def_readwrite("name", &PyAttachTarget::name, "Element name; ignored for World targets.");

    py::class_<PyAttachmentSpec>(
      m, "AttachmentSpec", "MJCF attachment applied to a robot root or a prior attachment."
    )
      .def(py::init<>())
      .def_readwrite("mjcf_path", &PyAttachmentSpec::mjcf_path, "Attachment MJCF path.")
      .def_readwrite(
        "attach_to", &PyAttachmentSpec::attach_to, "Parent element for this attachment."
      )
      .def_readwrite(
        "prefix", &PyAttachmentSpec::prefix, "Name prefix to avoid model element collisions."
      )
      .def_readwrite(
        "pos", &PyAttachmentSpec::pos, "Position offset in the parent frame, in meters."
      )
      .def_readwrite("euler", &PyAttachmentSpec::euler, "Extrinsic XYZ Euler offset, in degrees.")
      .def_readwrite(
        "contact_exclusions",
        &PyAttachmentSpec::contact_exclusions,
        "Body-name pairs registered as MuJoCo contact exclusions."
      );

    py::class_<PyRobotSpec>(
      m, "RobotSpec", "Robot root MJCF plus placement and ordered attachment specs."
    )
      .def(py::init<>())
      .def_readwrite("path", &PyRobotSpec::path, "Root robot MJCF path.")
      .def_readwrite("prefix", &PyRobotSpec::prefix, "Name prefix for multi-robot scenes.")
      .def_readwrite("attach_to", &PyRobotSpec::attach_to, "Placement parent; defaults to world.")
      .def_readwrite("pos", &PyRobotSpec::pos, "Placement offset in the parent frame, in meters.")
      .def_readwrite(
        "euler", &PyRobotSpec::euler, "Placement extrinsic XYZ Euler offset, in degrees."
      )
      .def_readwrite("attachments", &PyRobotSpec::attachments, "Ordered attachment chain.");

    py::class_<PySceneObject>(
      m,
      "SceneObject",
      "MJCF asset or primitive object placed in a scene. Primitive size, rgba, mass, and friction "
      "are required."
    )
      .def(py::init<>())
      .def_readwrite("name", &PySceneObject::name, "Object name.")
      .def_readwrite("mjcf_path", &PySceneObject::mjcf_path, "Optional MJCF asset path.")
      .def_readwrite("attach_to", &PySceneObject::attach_to, "Placement parent; defaults to world.")
      .def_readwrite("shape", &PySceneObject::shape, "Primitive shape when mjcf_path is empty.")
      .def_readwrite("size", &PySceneObject::size, "Required primitive size.")
      .def_readwrite("pos", &PySceneObject::pos, "Placement offset in the parent frame, in meters.")
      .def_readwrite("rgba", &PySceneObject::rgba, "Required primitive color.")
      .def_readwrite(
        "fixed", &PySceneObject::fixed, "If true, primitives are welded to their parent."
      )
      .def_readwrite("mass", &PySceneObject::mass, "Required primitive mass in kilograms.")
      .def_readwrite("condim", &PySceneObject::condim, "MuJoCo contact friction dimensionality.")
      .def_readwrite(
        "friction", &PySceneObject::friction, "Required primitive [slide, spin, roll] friction."
      );

    py::class_<PyCameraSpec>(
      m, "CameraSpec", "Named fixed world camera. pos and fovy are required."
    )
      .def(py::init<>())
      .def_readwrite("name", &PyCameraSpec::name, "Camera name.")
      .def_readwrite("pos", &PyCameraSpec::pos, "Required world position, in meters.")
      .def_readwrite("euler", &PyCameraSpec::euler, "Extrinsic XYZ Euler orientation, in degrees.")
      .def_readwrite("fovy", &PyCameraSpec::fovy, "Required vertical field of view, in degrees.");

    py::class_<PySceneSpec>(
      m, "SceneSpec", "Full scene description. timestep, add_floor, and add_skybox are required."
    )
      .def(py::init<>())
      .def_readwrite("robots", &PySceneSpec::robots, "Robot specs to add to the scene.")
      .def_readwrite(
        "timestep", &PySceneSpec::timestep, "Required MuJoCo physics timestep in seconds."
      )
      .def_readwrite(
        "gravity_z", &PySceneSpec::gravity_z, "World z gravity in m/s^2; defaults to -9.81."
      )
      .def_readwrite(
        "add_floor", &PySceneSpec::add_floor, "Required flag for checker ground plane."
      )
      .def_readwrite("add_skybox", &PySceneSpec::add_skybox, "Required flag for skybox and light.")
      .def_readwrite("objects", &PySceneSpec::objects, "Scene objects and fixtures.")
      .def_readwrite("cameras", &PySceneSpec::cameras, "Named fixed world cameras.");

    py::class_<PyToolFrameSpec>(
      m, "ToolFrameSpec", "Optional tool mass/TCP description for building the robot KDL chain."
    )
      .def(py::init<>())
      .def_readwrite(
        "tool_body", &PyToolFrameSpec::tool_body, "Root body of tool subtree to lump into dynamics."
      )
      .def_readwrite(
        "tcp_site", &PyToolFrameSpec::tcp_site, "MuJoCo site used as the terminal TCP frame."
      );

    py::class_<mj_kdl::ResetOptions>(m, "ResetOptions")
      .def(py::init<>())
      .def_readwrite("keyframe", &mj_kdl::ResetOptions::keyframe)
      .def_readwrite("use_keyframe", &mj_kdl::ResetOptions::use_keyframe);

    py::class_<mj_kdl::ResetInfo>(m, "ResetInfo")
      .def_readonly("used_keyframe", &mj_kdl::ResetInfo::used_keyframe)
      .def_readonly("keyframe", &mj_kdl::ResetInfo::keyframe);

    py::class_<PyScene, std::shared_ptr<PyScene>>(
      m, "Scene", "Owned MuJoCo model/data built from SceneSpec."
    )
      .def_static("build", &PyScene::build, py::arg("spec"), "Build and compile a MuJoCo scene.")
      .def("close", &PyScene::close, "Destroy owned MuJoCo model/data.")
      .def(
        "save_xml",
        &PyScene::save_xml,
        py::arg("path"),
        "Save the current model as XML when MuJoCo supports it."
      )
      .def(
        "save_binary",
        &PyScene::save_binary,
        py::arg("path"),
        "Save the current model as an .mjb binary."
      )
      .def("time", &PyScene::time, "Return current simulation time in seconds.")
      .def("timestep", &PyScene::timestep, "Return model physics timestep in seconds.")
      .def("step", &PyScene::step, "Advance MuJoCo by one step.")
      .def("step_n", &PyScene::step_n, py::arg("n"), "Advance MuJoCo by n steps.")
      .def("camera_names", &PyScene::camera_names, "Return names of cameras in the compiled model.")
      .def(
        "body_frame",
        &PyScene::body_frame,
        py::arg("name"),
        "Return a body world pose as PyKDL.Frame."
      )
      .def(
        "site_frame",
        &PyScene::site_frame,
        py::arg("name"),
        "Return a site world pose as PyKDL.Frame."
      )
      .def(
        "set_body_pose",
        [](
          PyScene                     &self,
          const std::string           &name,
          const std::array<double, 3> &pos,
          const py::object            &quat
        ) {
            if (quat.is_none()) {
                self.set_body_pose(name, pos, nullptr);
                return;
            }
            auto q = quat.cast<std::array<double, 4>>();
            std::array<double, 4> mj_quat = { q[3], q[0], q[1], q[2] };
            self.set_body_pose(name, pos, &mj_quat);
        },
        py::arg("name"),
        py::arg("pos"),
        py::arg("quat") = py::none(),
        "Set a body pose by name. quat is optional xyzw."
      )
      .def(
        "set_actuator_ctrl",
        &PyScene::set_actuator_ctrl,
        py::arg("name"),
        py::arg("value"),
        "Set an actuator ctrl slot by name."
      )
      .def(
        "actuator_ctrl",
        &PyScene::actuator_ctrl,
        py::arg("name"),
        "Read an actuator ctrl slot by name."
      )
      .def(
        "has_actuator",
        &PyScene::has_actuator,
        py::arg("name"),
        "Return true if an actuator exists in the compiled model."
      )
      .def(
        "add_object",
        &PyScene::add_object,
        py::arg("object"),
        "Rebuild the scene with an added object and rebind existing Robot handles."
      )
      .def(
        "remove_object",
        &PyScene::remove_object,
        py::arg("name"),
        "Rebuild the scene without the named object and rebind existing Robot handles."
      )
      .def_readonly("spec", &PyScene::spec, "Python scene spec used to build this scene.");

    py::class_<PyRobot, std::shared_ptr<PyRobot>>(
      m,
      "Robot",
      "Robot handle synchronized with a Scene or Env and backed by a wrapper-built KDL chain."
    )
      .def_static(
        "from_scene",
        [](
          const std::shared_ptr<PyScene> &scene,
          const std::string              &base_body,
          const std::string              &tip_body,
          const std::string              &prefix,
          const py::object               &tool
        ) {
            if (tool.is_none()) {
                return PyRobot::from_scene(scene, base_body, tip_body, prefix, nullptr);
            }
            auto cpp_tool = tool.cast<PyToolFrameSpec>();
            return PyRobot::from_scene(scene, base_body, tip_body, prefix, &cpp_tool);
        },
        py::arg("scene"),
        py::arg("base_body"),
        py::arg("tip_body"),
        py::arg("prefix") = "",
        py::arg("tool")   = py::none(),
        "Create a robot handle from a Scene and build its KDL chain."
      )
      .def("update", &PyRobot::update, "Synchronize measured joint state from MuJoCo.")
      .def("step", &PyRobot::step, "Apply commands and step the robot's MuJoCo scene.")
      .def("step_n", &PyRobot::step_n, py::arg("n"), "Run step() n times.")
      .def(
        "set_joint_pos",
        [](PyRobot &self, const py::object &q, bool call_forward) {
            self.set_joint_pos(to_jnt_array(q, self.robot.n_joints), call_forward);
        },
        py::arg("q"),
        py::arg("call_forward") = true,
        "Set measured MuJoCo joint positions."
      )
      .def(
        "gravity_torques",
        &PyRobot::gravity_torques,
        py::arg("gravity_z") = -9.81,
        "Compute KDL gravity torques at measured q."
      )
      .def("kdl_chain", &PyRobot::kdl_chain, "Return the wrapper-built chain as a PyKDL.Chain.")
      .def(
        "fk_frame",
        [](const PyRobot &self, const py::object &q) {
            if (q.is_none()) return self.fk_frame(nullptr);
            KDL::JntArray values = to_jnt_array(q, self.robot.n_joints);
            return self.fk_frame(&values);
        },
        py::arg("q") = py::none(),
        "Return FK terminal pose as PyKDL.Frame for measured q or the provided q."
      )
      .def_property(
        "ctrl_mode",
        [](const PyRobot &self) {
            self.ensure_active();
            return self.robot.ctrl_mode;
        },
        [](PyRobot &self, CtrlMode mode) {
            self.ensure_active();
            self.robot.ctrl_mode = mode;
        }
      )
      .def_property(
        "paused",
        [](const PyRobot &self) {
            self.ensure_active();
            return self.robot.paused;
        },
        [](PyRobot &self, bool paused) {
            self.ensure_active();
            self.robot.paused = paused;
        }
      )
      .def_property_readonly(
        "n_joints",
        [](const PyRobot &self) {
            self.ensure_active();
            return self.robot.n_joints;
        }
      )
      .def_property_readonly(
        "joint_names",
        [](const PyRobot &self) {
            self.ensure_active();
            return self.robot.joint_names;
        }
      )
      .def_property_readonly("joint_limits", &PyRobot::joint_limits)
      .def_property(
        "jnt_pos_msr",
        [](const PyRobot &self) {
            self.ensure_active();
            return self.robot.jnt_pos_msr;
        },
        [](PyRobot &self, const std::vector<double> &values) {
            self.set_port(&mj_kdl::Robot::jnt_pos_msr, values);
        }
      )
      .def_property(
        "jnt_vel_msr",
        [](const PyRobot &self) {
            self.ensure_active();
            return self.robot.jnt_vel_msr;
        },
        [](PyRobot &self, const std::vector<double> &values) {
            self.set_port(&mj_kdl::Robot::jnt_vel_msr, values);
        }
      )
      .def_property(
        "jnt_trq_msr",
        [](const PyRobot &self) {
            self.ensure_active();
            return self.robot.jnt_trq_msr;
        },
        [](PyRobot &self, const std::vector<double> &values) {
            self.set_port(&mj_kdl::Robot::jnt_trq_msr, values);
        }
      )
      .def_property(
        "jnt_pos_cmd",
        [](const PyRobot &self) {
            self.ensure_active();
            return self.robot.jnt_pos_cmd;
        },
        [](PyRobot &self, const std::vector<double> &values) {
            self.set_port(&mj_kdl::Robot::jnt_pos_cmd, values);
        }
      )
      .def_property(
        "jnt_trq_cmd",
        [](const PyRobot &self) {
            self.ensure_active();
            return self.robot.jnt_trq_cmd;
        },
        [](PyRobot &self, const std::vector<double> &values) {
            self.set_port(&mj_kdl::Robot::jnt_trq_cmd, values);
        }
      );

    py::class_<PySimulateViewer, std::shared_ptr<PySimulateViewer>>(
      m, "SimulateViewer", "Wrapper for the custom MuJoCo simulate UI."
    )
      .def_static(
        "open",
        &PySimulateViewer::open,
        py::arg("robot"),
        py::arg("title") = "MuJoCo",
        "Open the custom simulate UI for a robot."
      )
      .def("close", &PySimulateViewer::close, "Close the viewer.")
      .def(
        "is_running", &PySimulateViewer::is_running, "Return false once the user closes the viewer."
      )
      .def("step", &PySimulateViewer::step, "Step simulation and update the viewer.")
      .def("step_n", &PySimulateViewer::step_n, py::arg("n"), "Run step() n times.")
      .def("clear_trace", &PySimulateViewer::clear_trace, "Clear viewer trace geometry.")
      .def(
        "add_trace_segment",
        [](
          PySimulateViewer            &self,
          const std::array<double, 3> &a,
          const std::array<double, 3> &b,
          const py::object            &rgba
        ) {
            if (rgba.is_none()) {
                self.add_trace_segment(a, b, nullptr);
                return;
            }
            auto color = rgba.cast<std::array<float, 4>>();
            self.add_trace_segment(a, b, &color);
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("rgba") = py::none(),
        "Add a trace line segment in world coordinates."
      )
      .def(
        "use_camera",
        &PySimulateViewer::use_camera,
        py::arg("name") = "",
        "Switch to a named camera; empty name restores default."
      )
      .def_property(
        "realtime_factor",
        [](const PySimulateViewer &self) { return self.viewer.realtime_factor; },
        [](PySimulateViewer &self, double value) {
            if (value < 0.0) throw std::invalid_argument("realtime_factor must be >= 0");
            self.viewer.realtime_factor = value;
        }
      );

    py::class_<PyVideoRecorder, std::shared_ptr<PyVideoRecorder>>(
      m, "VideoRecorder", "Offscreen MuJoCo video recorder for a Scene."
    )
      .def_static(
        "open",
        &PyVideoRecorder::open,
        py::arg("scene"),
        py::arg("out_path"),
        py::arg("width")  = 1280,
        py::arg("height") = 720,
        py::arg("fps")    = 60,
        "Open an offscreen video recorder with explicit dimensions."
      )
      .def_static(
        "open_preset",
        &PyVideoRecorder::open_preset,
        py::arg("scene"),
        py::arg("out_path"),
        py::arg("resolution") = mj_kdl::VideoResolution::R720p,
        py::arg("fps")        = 60,
        "Open an offscreen video recorder with a resolution preset."
      )
      .def("record_frame", &PyVideoRecorder::record_frame, "Render and append one frame.")
      .def(
        "use_camera",
        &PyVideoRecorder::use_camera,
        py::arg("name") = "",
        "Switch to a named camera; empty name restores default."
      )
      .def("close", &PyVideoRecorder::close, "Finalize and close the recorder.");

    py::class_<PyEnv, std::shared_ptr<PyEnv>>(
      m, "Env", "Resettable scene environment that keeps registered Robot handles synchronized."
    )
      .def_static(
        "build", &PyEnv::build, py::arg("spec"), "Build a resettable environment from a SceneSpec."
      )
      .def("close", &PyEnv::close, "Destroy owned MuJoCo resources.")
      .def(
        "create_robot",
        [](
          const std::shared_ptr<PyEnv> &self,
          const std::string            &base_body,
          const std::string            &tip_body,
          const std::string            &prefix,
          const py::object             &tool
        ) {
            if (tool.is_none()) return self->create_robot(base_body, tip_body, prefix, nullptr);
            auto cpp_tool = tool.cast<PyToolFrameSpec>();
            return self->create_robot(base_body, tip_body, prefix, &cpp_tool);
        },
        py::arg("base_body"),
        py::arg("tip_body"),
        py::arg("prefix") = "",
        py::arg("tool")   = py::none(),
        "Create and register a robot handle in this environment."
      )
      .def(
        "reset",
        [](PyEnv &self, const py::object &options) {
            if (options.is_none()) return self.reset(nullptr);
            auto cpp_options = options.cast<mj_kdl::ResetOptions>();
            return self.reset(&cpp_options);
        },
        py::arg("options") = py::none(),
        "Reset MuJoCo state and synchronize registered robots."
      )
      .def(
        "add_object",
        &PyEnv::add_object,
        py::arg("object"),
        "Rebuild the environment with an added object and rebind existing Robot handles."
      )
      .def(
        "remove_object",
        &PyEnv::remove_object,
        py::arg("name"),
        "Rebuild the environment without the named object and rebind existing Robot handles."
      )
      .def("camera_names", &PyEnv::camera_names, "Return names of cameras in the compiled model.")
      .def_readonly("spec", &PyEnv::spec, "Python scene spec used to build this environment.");

    m.def("set_log_level", &mj_kdl::set_log_level, py::arg("level"), "Set wrapper log level.");
    m.def("get_log_level", &mj_kdl::get_log_level, "Return wrapper log level.");
    m.def(
      "mujoco_version",
      []() { return std::string(mj_versionString()); },
      "Return the MuJoCo version used by the extension."
    );
    m.def(
      "scene_object_site_name",
      [](const PySceneObject &obj, const std::string &site_name) {
          SceneObject cpp_obj = to_cpp(obj);
          return mj_kdl::scene_object_site_name(cpp_obj, site_name.c_str());
      },
      py::arg("object"),
      py::arg("site_name"),
      "Return the compiled site name for a SceneObject-authored site."
    );
}
