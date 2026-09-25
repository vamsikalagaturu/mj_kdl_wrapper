#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <mujoco/mujoco.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

// Version is injected by CMake from the single source of truth (pyproject.toml).
#ifndef MJ_KDL_WRAPPER_VERSION
#define MJ_KDL_WRAPPER_VERSION "0.0.0+unknown"
#endif

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
    std::array<double, 3>                            pos  = { 0.0, 0.0, 0.0 };
    std::array<double, 4>                            quat = { 0.0, 0.0, 0.0, 1.0 };
    std::vector<std::pair<std::string, std::string>> contact_exclusions;
};

struct PyRobotSpec
{
    std::string                   path;
    std::string                   prefix;
    PyAttachTarget                attach_to;
    std::array<double, 3>         pos  = { 0.0, 0.0, 0.0 };
    std::array<double, 4>         quat = { 0.0, 0.0, 0.0, 1.0 };
    std::vector<PyAttachmentSpec> attachments;
    std::vector<mj_kdl::CtrlModeSpec> modes = mj_kdl::RobotSpec{}.modes;
};

struct PySceneObject
{
    std::string                          name;
    std::string                          mjcf_path;
    std::string                          prefix;
    PyAttachTarget                       attach_to;
    Shape                                shape = Shape::Unspecified;
    std::optional<std::array<double, 3>> size;
    std::array<double, 3>                pos  = { 0.0, 0.0, 0.0 };
    std::array<double, 4>                quat = { 0.0, 0.0, 0.0, 1.0 };
    std::optional<std::array<float, 4>>  rgba;
    bool                                 fixed = false;
    std::optional<double>                mass;
    Condim                               condim = Condim::Tangential;
    std::optional<std::array<double, 3>> friction;
};

struct PyCameraSpec
{
    std::string                          name;
    std::string                          body;
    std::optional<std::array<double, 3>> pos;
    std::array<double, 4>                quat = { 0.0, 0.0, 0.0, 1.0 };
    std::optional<double>                fovy;
};

struct PySiteSpec
{
    std::string           body;
    std::string           name;
    std::array<double, 3> pos  = { 0.0, 0.0, 0.0 };
    std::array<double, 4> quat = { 0.0, 0.0, 0.0, 1.0 };
};

struct PySceneSpec
{
    std::vector<PyRobotSpec>   robots;
    std::optional<double>      timestep;
    double                     gravity_z = -9.81;
    std::optional<bool>        add_floor;
    double                     floor_z = 0.0;
    std::optional<bool>        add_skybox;
    std::vector<PySceneObject> objects;
    std::vector<PySiteSpec>    sites;
    std::vector<PyCameraSpec>  cameras;
};

struct PyForceTorqueSensorSpec
{
    std::string name;
    std::string force_sensor;
    std::string torque_sensor;
    std::string frame_site;
};

struct PyToolFrameSpec
{
    std::string                          tool_body;
    std::string                          tcp_site;
    std::vector<PyForceTorqueSensorSpec> ft_sensors;
};

struct PyEnv;
struct PyRobot;

AttachTarget to_cpp(const PyAttachTarget &src)
{
    AttachTarget out;
    out.kind = src.kind;
    if (src.kind != AttachKind::World) out.name = src.name;
    return out;
}

mj_kdl::AttachmentSpec to_cpp(const PyAttachmentSpec &src)
{
    mj_kdl::AttachmentSpec out;
    out.mjcf_path = src.mjcf_path;
    out.attach_to = to_cpp(src.attach_to);
    out.prefix    = src.prefix;
    std::copy(src.pos.begin(), src.pos.end(), out.pos);
    std::copy(src.quat.begin(), src.quat.end(), out.quat);
    out.contact_exclusions = src.contact_exclusions;
    return out;
}

RobotSpec to_cpp(const PyRobotSpec &src)
{
    RobotSpec out;
    out.path      = src.path;
    out.prefix    = src.prefix;
    out.attach_to = to_cpp(src.attach_to);
    std::copy(src.pos.begin(), src.pos.end(), out.pos);
    std::copy(src.quat.begin(), src.quat.end(), out.quat);
    out.attachments.reserve(src.attachments.size());
    for (const auto &item : src.attachments) out.attachments.push_back(to_cpp(item));
    out.modes = src.modes;
    return out;
}

SceneObject to_cpp(const PySceneObject &src)
{
    SceneObject out;
    out.name      = src.name;
    out.mjcf_path = src.mjcf_path;
    out.prefix    = src.prefix;
    out.attach_to = to_cpp(src.attach_to);
    out.shape     = src.shape;
    std::copy(src.pos.begin(), src.pos.end(), out.pos);
    std::copy(src.quat.begin(), src.quat.end(), out.quat);
    out.fixed  = src.fixed;
    out.condim = src.condim;
    if (src.rgba) {
        std::copy(src.rgba->begin(), src.rgba->end(), out.rgba);
        out.has_rgba = true;
    }
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
        out.mass = *src.mass;
        std::copy(src.friction->begin(), src.friction->end(), out.friction);
    }
    return out;
}

CameraSpec to_cpp(const PyCameraSpec &src)
{
    CameraSpec out;
    out.name = src.name;
    out.body = src.body;
    if (!src.pos) throw std::runtime_error("CameraSpec.pos must be set");
    if (!src.fovy) throw std::runtime_error("CameraSpec.fovy must be set");
    std::copy(src.pos->begin(), src.pos->end(), out.pos);
    std::copy(src.quat.begin(), src.quat.end(), out.quat);
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
    out.floor_z    = src.floor_z;
    out.add_skybox = *src.add_skybox;
    out.robots.reserve(src.robots.size());
    for (const auto &item : src.robots) out.robots.push_back(to_cpp(item));
    out.objects.reserve(src.objects.size());
    for (const auto &item : src.objects) out.objects.push_back(to_cpp(item));
    out.sites.reserve(src.sites.size());
    for (const auto &item : src.sites) {
        mj_kdl::SiteSpec site;
        site.body = item.body;
        site.name = item.name;
        std::copy(item.pos.begin(), item.pos.end(), site.pos);
        std::copy(item.quat.begin(), item.quat.end(), site.quat);
        out.sites.push_back(site);
    }
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
              "expected " + std::to_string(expected) + " joint values, got " + std::to_string(rows)
            );
        }
        KDL::JntArray out(expected);
        for (int i = 0; i < expected; ++i) out(i) = values[py::int_(i)].cast<double>();
        return out;
    }
    return to_jnt_array(values.cast<std::vector<double>>(), expected);
}

// A copy, not a view: reset() replaces the port buffers a view would point into.
template<typename T, typename Values> py::array_t<T> read_only_array(const Values &values)
{
    py::array_t<T> out(static_cast<py::ssize_t>(values.size()));
    std::copy(values.begin(), values.end(), out.mutable_data());
    out.attr("setflags")(py::arg("write") = false);
    return out;
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

py::object kdl_wrench_to_py(const KDL::Wrench &wrench)
{
    py::module_ kdl   = py::module_::import("PyKDL");
    py::object  force = kdl.attr("Vector")(wrench.force.x(), wrench.force.y(), wrench.force.z());
    py::object torque = kdl.attr("Vector")(wrench.torque.x(), wrench.torque.y(), wrench.torque.z());
    return kdl.attr("Wrench")(force, torque);
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

mj_kdl::ToolFrameSpec to_cpp(const PyToolFrameSpec &src)
{
    mj_kdl::ToolFrameSpec out;
    out.tool_body = src.tool_body;
    out.tcp_site  = src.tcp_site;
    out.ft_sensors.reserve(src.ft_sensors.size());
    for (const auto &item : src.ft_sensors) {
        out.ft_sensors.push_back(mj_kdl::ForceTorqueSensorSpec{
          .name          = item.name,
          .force_sensor  = item.force_sensor,
          .torque_sensor = item.torque_sensor,
          .frame_site    = item.frame_site,
        });
    }
    return out;
}

struct PyEnv : std::enable_shared_from_this<PyEnv>
{
    PySceneSpec spec;
    mj_kdl::Env env;
    py::object  reset_callback;        // Python callable invoked by reset(), or None
    py::object  model_obj, data_obj;   // mujoco.MjModel / MjData the Env runs on
    py::object  next_model, next_data; // made by adopt during a (re)build, taken after it

    ~PyEnv() { close(); }

    void ensure_open() const
    {
        if (!env.model || !env.data) throw std::runtime_error("env is closed");
    }

    // init_env clears Env::on_reset, so the hook is wired after it.
    void wire_reset_hook()
    {
        env.on_reset = [this](mj_kdl::ResetContext *ctx) {
            if (!reset_callback || reset_callback.is_none()) return;
            py::gil_scoped_acquire gil;
            reset_callback(ctx);
        };
    }

    // The Env runs on mujoco-owned objects, so Python can call the mujoco API on the live state.
    void wire_adopt()
    {
        env.adopt = [this](mjModel *m, mjData *d) {
            py::gil_scoped_acquire gil;
            const int              size = mj_sizeModel(m);
            std::string            mjb(static_cast<size_t>(size), '\0');
            mj_saveModel(m, nullptr, mjb.data(), size);
            py::module_ mujoco = py::module_::import("mujoco");
            py::dict    assets;
            assets["scene.mjb"] = py::bytes(mjb);
            next_model = mujoco.attr("MjModel").attr("from_binary_path")("scene.mjb", assets);
            next_data  = mujoco.attr("MjData")(next_model);
            auto *am =
              reinterpret_cast<mjModel *>(next_model.attr("_address").cast<std::uintptr_t>());
            auto *ad =
              reinterpret_cast<mjData *>(next_data.attr("_address").cast<std::uintptr_t>());
            // An .mjb drops the compiler signature, which save_model_xml's mj_copyBack checks.
            am->signature = m->signature;
            mj_copyData(ad, m, d);
            mj_kdl::destroy_scene(m, d);
            return std::pair{ am, ad };
        };
    }

    // After a (re)build returns, the new pair is in use and the old one can go.
    void take_next()
    {
        py::gil_scoped_acquire gil;
        model_obj = std::move(next_model);
        data_obj  = std::move(next_data);
    }

    static std::shared_ptr<PyEnv> build(const PySceneSpec &spec)
    {
        auto out           = std::shared_ptr<PyEnv>(new PyEnv());
        out->spec          = spec;
        SceneSpec cpp_spec = to_cpp(out->spec);
        out->wire_adopt();
        mj_kdl::Status s;
        {
            py::gil_scoped_release nogil;
            s = mj_kdl::init_env(&out->env, &cpp_spec);
        }
        if (!s) throw std::runtime_error(s.error);
        out->take_next();
        out->wire_reset_hook();
        return out;
    }

    // The viewer thread is stopped by cleanup before the mujoco objects are dropped.
    void close()
    {
        mj_kdl::cleanup(&env);
        model_obj = py::object();
        data_obj  = py::object();
    }

    std::shared_ptr<PyRobot> create_robot(
      const std::string     &base_body,
      const std::string     &tip_body,
      const std::string     &prefix,
      const PyToolFrameSpec *tool
    );

    std::shared_ptr<PyRobot> create_robot_from_chain(
      const KDL::Chain               &chain,
      const std::vector<std::string> &joint_names,
      const std::string              &prefix,
      const PyToolFrameSpec          *tool
    );

    mj_kdl::ResetInfo reset(const mj_kdl::ResetOptions *options)
    {
        ensure_open();
        return mj_kdl::reset(&env, options);
    }

    bool step()
    {
        ensure_open();
        return mj_kdl::step(&env);
    }

    void update()
    {
        ensure_open();
        mj_kdl::update(&env);
    }

    void pace()
    {
        ensure_open();
        mj_kdl::pace_realtime(&env);
    }

    void open_viewer(const std::string &title)
    {
        ensure_open();
        if (mj_kdl::Status s = mj_kdl::open_viewer(&env, title.c_str()); !s)
            throw std::runtime_error(s.error);
    }

    void add_object(const PySceneObject &object)
    {
        ensure_open();
        if (mj_kdl::Status s = mj_kdl::scene_add_object(&env, to_cpp(object)); !s)
            throw std::runtime_error(s.error);
        take_next();
        spec.objects.push_back(object);
    }

    void remove_object(const std::string &name)
    {
        ensure_open();
        auto it = std::find_if(spec.objects.begin(), spec.objects.end(), [&](const auto &obj) {
            return obj.name == name;
        });
        if (it == spec.objects.end()) throw std::runtime_error("object not found");
        if (mj_kdl::Status s = mj_kdl::scene_remove_object(&env, name); !s)
            throw std::runtime_error(s.error);
        take_next();
        spec.objects.erase(it);
    }

    void set_control_mode(int robot, CtrlMode mode)
    {
        ensure_open();
        if (mj_kdl::Status s = mj_kdl::set_control_mode(&env, robot, mode); !s)
            throw std::runtime_error(s.error);
    }

    py::object body_frame(const std::string &name)
    {
        ensure_open();
        KDL::Frame frame;
        if (!mj_kdl::get_body_frame(&env, name.c_str(), &frame)) {
            throw std::runtime_error("body not found");
        }
        return kdl_frame_to_py(frame);
    }

    py::object site_frame(const std::string &name)
    {
        ensure_open();
        KDL::Frame frame;
        if (!mj_kdl::get_site_frame(&env, name.c_str(), &frame)) {
            throw std::runtime_error("site not found");
        }
        return kdl_frame_to_py(frame);
    }

    void set_body_pose(
      const std::string           &name,
      const std::array<double, 3> &pos,
      const std::array<double, 4> *quat_xyzw
    )
    {
        ensure_open();
        mj_kdl::set_body_pose(
          &env, name.c_str(), pos.data(), quat_xyzw ? quat_xyzw->data() : nullptr
        );
    }

    void save_model_xml(const std::string &path) const
    {
        ensure_open();
        if (mj_kdl::Status s = mj_kdl::save_model_xml(env.model, path.c_str()); !s)
            throw std::runtime_error(s.error);
    }
};

struct PyRobot
{
    // Declared before robot so the robot unregisters while its Env still exists.
    std::shared_ptr<PyEnv> env_owner;
    mj_kdl::Robot          robot;

    void ensure_active() const
    {
        if (!robot.model || !robot.data || robot.n_joints <= 0) {
            throw std::runtime_error("robot is closed");
        }
    }

    void set_joint_pos(const KDL::JntArray &joints)
    {
        ensure_active();
        if (static_cast<int>(joints.rows()) != robot.n_joints) {
            throw std::invalid_argument(
              "expected " + std::to_string(robot.n_joints) + " joint values, got "
              + std::to_string(joints.rows())
            );
        }
        mj_kdl::set_joint_pos(&robot, joints);
    }

    std::vector<std::pair<double, double>> joint_limits() const
    {
        ensure_active();
        return robot.joint_limits;
    }

    py::object kdl_chain() const
    {
        ensure_active();
        return kdl_chain_to_py(robot.chain);
    }

    std::vector<std::string> ft_sensor_names() const
    {
        ensure_active();
        std::vector<std::string> out;
        out.reserve(robot.ft_sensors.size());
        for (const auto &sensor : robot.ft_sensors) out.push_back(sensor.name);
        return out;
    }

    py::object ft_sensor(const std::string &name) const
    {
        ensure_active();
        for (const auto &sensor : robot.ft_sensors)
            if (sensor.name == name) return kdl_wrench_to_py(sensor.wrench);
        throw std::runtime_error("FT sensor not found: " + name);
    }

    void set_port(std::vector<double> mj_kdl::RobotPorts::*port, const std::vector<double> &values)
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

std::shared_ptr<PyRobot> PyEnv::create_robot(
  const std::string     &base_body,
  const std::string     &tip_body,
  const std::string     &prefix,
  const PyToolFrameSpec *tool
)
{
    ensure_open();
    auto out       = std::make_shared<PyRobot>();
    out->env_owner = shared_from_this();
    mj_kdl::ToolFrameSpec cpp_tool;
    if (tool) cpp_tool = to_cpp(*tool);
    const mj_kdl::Status s = mj_kdl::init_robot_from_mjcf(
      &out->robot,
      &env,
      base_body.c_str(),
      tip_body.c_str(),
      prefix.c_str(),
      tool ? &cpp_tool : nullptr
    );
    if (!s) throw std::runtime_error(s.error);
    return out;
}

std::shared_ptr<PyRobot> PyEnv::create_robot_from_chain(
  const KDL::Chain               &chain,
  const std::vector<std::string> &joint_names,
  const std::string              &prefix,
  const PyToolFrameSpec          *tool
)
{
    ensure_open();
    auto out       = std::make_shared<PyRobot>();
    out->env_owner = shared_from_this();
    mj_kdl::ToolFrameSpec cpp_tool;
    if (tool) cpp_tool = to_cpp(*tool);
    const mj_kdl::Status s = mj_kdl::init_robot_from_chain(
      &out->robot, &env, chain, joint_names, prefix.c_str(), tool ? &cpp_tool : nullptr
    );
    if (!s) throw std::runtime_error(s.error);
    return out;
}

// The simulate UI of an Env; every call is a no-op (or false) while it is closed.
struct PyViewer
{
    std::shared_ptr<PyEnv> owner;

    mj_kdl::Viewer *viewer() const { return &owner->env.viewer; }

    bool is_running() const { return mj_kdl::is_running(viewer()); }

    bool key_pressed(int glfw_key) const { return mj_kdl::key_pressed(viewer(), glfw_key); }

    void capture_key(int glfw_key, bool capture)
    {
        mj_kdl::capture_key(viewer(), glfw_key, capture);
    }

    void clear_trace() { mj_kdl::clear_trace(viewer()); }

    void add_trace_segment(
      const std::array<double, 3> &a,
      const std::array<double, 3> &b,
      const std::array<float, 4>  *rgba
    )
    {
        KDL::Vector ka(a[0], a[1], a[2]);
        KDL::Vector kb(b[0], b[1], b[2]);
        mj_kdl::add_trace_segment(viewer(), ka, kb, rgba ? rgba->data() : nullptr);
    }

    bool use_camera(const std::string &name)
    {
        owner->ensure_open();
        return mj_kdl::use_camera(
          viewer(), owner->env.model, name.empty() ? nullptr : name.c_str()
        );
    }

    void set_free_camera(
      double                       distance,
      double                       azimuth,
      double                       elevation,
      const std::array<double, 3> &lookat
    )
    {
        mj_kdl::set_free_camera(viewer(), distance, azimuth, elevation, lookat);
    }
};

struct PyVideoRecorder
{
    mj_kdl::VideoRecorder  recorder;
    std::shared_ptr<PyEnv> owner; // keeps the Env alive for the recorder's lifetime
    bool                   active = false;
    int                    width  = 0; // 0 for a preset recorder
    int                    height = 0;

    ~PyVideoRecorder() { close(); }

    // width <= 0 selects the preset.
    static std::shared_ptr<PyVideoRecorder> make(
      const std::shared_ptr<PyEnv> &env,
      const std::string            &out_path,
      int                           width,
      int                           height,
      mj_kdl::VideoResolution       resolution,
      bool                          use_preset,
      int                           fps
    )
    {
        if (!env) throw std::runtime_error("env is null");
        env->ensure_open();
        mjModel *m   = env->env.model;
        auto     out = std::shared_ptr<PyVideoRecorder>(new PyVideoRecorder());
        out->owner   = env;
        const mj_kdl::Status s =
          use_preset
            ? mj_kdl::init_video_recorder(&out->recorder, m, out_path.c_str(), resolution, fps)
            : mj_kdl::init_video_recorder(&out->recorder, m, out_path.c_str(), width, height, fps);
        if (!s) throw std::runtime_error(s.error);
        out->active = true;
        if (!use_preset) {
            out->width  = width;
            out->height = height;
        }
        return out;
    }

    static std::shared_ptr<PyVideoRecorder>
      make_offscreen(const std::shared_ptr<PyEnv> &env, int width, int height)
    {
        if (!env) throw std::runtime_error("env is null");
        env->ensure_open();
        auto out   = std::shared_ptr<PyVideoRecorder>(new PyVideoRecorder());
        out->owner = env;
        if (mj_kdl::Status s =
              mj_kdl::init_offscreen(&out->recorder, env->env.model, width, height);
            !s)
            throw std::runtime_error(s.error);
        out->active = true;
        out->width  = width;
        out->height = height;
        return out;
    }

    py::array_t<std::uint8_t> render_rgb()
    {
        if (!active || !owner->env.model) throw std::runtime_error("recorder is closed");
        if (width <= 0) {
            throw std::runtime_error(
              "render_rgb needs the frame size: open the recorder with width and height"
            );
        }
        py::array_t<std::uint8_t> out({ height, width, 3 });
        std::uint8_t             *pixels = out.mutable_data();
        bool                      ok     = false;
        {
            py::gil_scoped_release nogil;
            ok = mj_kdl::render_rgb(&recorder, &owner->env, pixels);
        }
        if (!ok) throw std::runtime_error("render_rgb failed");
        return out;
    }

    bool record_frame()
    {
        if (!active || !owner->env.model) throw std::runtime_error("recorder is closed");
        return mj_kdl::record_frame(&recorder, &owner->env);
    }

    bool use_camera(const std::string &name)
    {
        if (!active || !owner->env.model) throw std::runtime_error("recorder is closed");
        if (name.empty()) {
            mjv_defaultFreeCamera(owner->env.model, &recorder.cam);
            return true;
        }
        const int id = mj_name2id(owner->env.model, mjOBJ_CAMERA, name.c_str());
        if (id < 0) return false;
        recorder.cam.type       = mjCAMERA_FIXED;
        recorder.cam.fixedcamid = id;
        return true;
    }

    void set_free_camera(
      double                       distance,
      double                       azimuth,
      double                       elevation,
      const std::array<double, 3> &lookat
    )
    {
        if (!active) throw std::runtime_error("recorder is closed");
        mjvCamera &cam = recorder.cam;
        cam.type       = mjCAMERA_FREE;
        cam.fixedcamid = -1;
        cam.distance   = distance;
        cam.azimuth    = azimuth;
        cam.elevation  = elevation;
        std::copy(lookat.begin(), lookat.end(), cam.lookat);
    }

    void close()
    {
        if (!active) return;
        mj_kdl::cleanup(&recorder);
        active = false;
        owner.reset();
    }
};

} // namespace

PYBIND11_MODULE(_mj_kdl_wrapper, m)
{
    m.doc()                      = "Python bindings for mj_kdl_wrapper";
    m.attr("__version__")        = MJ_KDL_WRAPPER_VERSION;
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
      .value("TORQUE", CtrlMode::TORQUE)
      .value("VELOCITY", CtrlMode::VELOCITY);

    py::class_<mj_kdl::CtrlModeSpec>(
      m, "CtrlModeSpec", "A control mode build_scene adds to a robot, on its own actuator group."
    )
      .def(
        py::init([](CtrlMode mode, std::vector<std::string> joints, double kv) {
            return mj_kdl::CtrlModeSpec{ mode, std::move(joints), kv };
        }),
        py::arg("mode")   = CtrlMode::TORQUE,
        py::arg("joints") = std::vector<std::string>{},
        py::arg("kv")     = 0.0
      )
      .def_readwrite("mode", &mj_kdl::CtrlModeSpec::mode)
      .def_readwrite(
        "joints", &mj_kdl::CtrlModeSpec::joints, "Joints to give the mode; empty = all actuated."
      )
      .def_readwrite("kv", &mj_kdl::CtrlModeSpec::kv, "VELOCITY actuator gain [N m s/rad].");

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
      .def_readwrite("quat", &PyAttachmentSpec::quat, "Orientation offset [x, y, z, w].")
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
      .def_readwrite("quat", &PyRobotSpec::quat, "Placement orientation offset [x, y, z, w].")
      .def_readwrite("attachments", &PyRobotSpec::attachments, "Ordered attachment chain.")
      .def_readwrite(
        "modes", &PyRobotSpec::modes, "Extra control modes; default [TORQUE], [] = native only."
      );

    py::class_<PySceneObject>(
      m,
      "SceneObject",
      "MJCF asset or primitive object placed in a scene. Primitive size, rgba, mass, and friction "
      "are required."
    )
      .def(py::init<>())
      .def_readwrite("name", &PySceneObject::name, "Object name.")
      .def_readwrite("mjcf_path", &PySceneObject::mjcf_path, "Optional MJCF asset path.")
      .def_readwrite(
        "prefix",
        &PySceneObject::prefix,
        "Prepended to the asset's element names; empty = as authored."
      )
      .def_readwrite("attach_to", &PySceneObject::attach_to, "Placement parent; defaults to world.")
      .def_readwrite("shape", &PySceneObject::shape, "Primitive shape when mjcf_path is empty.")
      .def_readwrite("size", &PySceneObject::size, "Required primitive size.")
      .def_readwrite("pos", &PySceneObject::pos, "Placement offset in the parent frame, in meters.")
      .def_readwrite("quat", &PySceneObject::quat, "Placement orientation offset [x, y, z, w].")
      .def_readwrite(
        "rgba", &PySceneObject::rgba, "Primitive color, required; on an asset, recolors its geoms."
      )
      .def_readwrite(
        "fixed", &PySceneObject::fixed, "If true, primitives are welded to their parent."
      )
      .def_readwrite("mass", &PySceneObject::mass, "Required primitive mass in kilograms.")
      .def_readwrite("condim", &PySceneObject::condim, "MuJoCo contact friction dimensionality.")
      .def_readwrite(
        "friction", &PySceneObject::friction, "Required primitive [slide, spin, roll] friction."
      );

    py::class_<PySiteSpec>(m, "SiteSpec", "A frame marked on a body of the assembled scene.")
      .def(py::init<>())
      .def_readwrite("body", &PySiteSpec::body, "Body to add the site to, by name.")
      .def_readwrite("name", &PySiteSpec::name, "Site name, unique within the scene.")
      .def_readwrite("pos", &PySiteSpec::pos, "Offset in the body frame, in meters.")
      .def_readwrite("quat", &PySiteSpec::quat, "Orientation in the body frame [x, y, z, w].");

    py::class_<PyCameraSpec>(
      m,
      "CameraSpec",
      "Named camera on a body, or in the world when body is empty. pos and fovy are required."
    )
      .def(py::init<>())
      .def_readwrite("name", &PyCameraSpec::name, "Camera name.")
      .def_readwrite("body", &PyCameraSpec::body, "Anchor body name; empty means the worldbody.")
      .def_readwrite(
        "pos", &PyCameraSpec::pos, "Required position in the anchor body's frame, in meters."
      )
      .def_readwrite("quat", &PyCameraSpec::quat, "Orientation [x, y, z, w].")
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
      .def_readwrite(
        "floor_z", &PySceneSpec::floor_z, "Ground plane height in the world frame; defaults to 0."
      )
      .def_readwrite("add_skybox", &PySceneSpec::add_skybox, "Required flag for skybox and light.")
      .def_readwrite("objects", &PySceneSpec::objects, "Scene objects and fixtures.")
      .def_readwrite("sites", &PySceneSpec::sites, "Frames to mark on the scene's bodies.")
      .def_readwrite("cameras", &PySceneSpec::cameras, "Named fixed world cameras.");

    py::class_<PyForceTorqueSensorSpec>(
      m, "ForceTorqueSensorSpec", "Logical FT sensor backed by MuJoCo force and torque sensors."
    )
      .def(py::init<>())
      .def_readwrite("name", &PyForceTorqueSensorSpec::name, "Logical wrapper sensor name.")
      .def_readwrite(
        "force_sensor",
        &PyForceTorqueSensorSpec::force_sensor,
        "MuJoCo force sensor name; defaults to '<name>_force'."
      )
      .def_readwrite(
        "torque_sensor",
        &PyForceTorqueSensorSpec::torque_sensor,
        "MuJoCo torque sensor name; defaults to '<name>_torque'."
      )
      .def_readwrite(
        "frame_site",
        &PyForceTorqueSensorSpec::frame_site,
        "Optional MuJoCo site that defines the sensor frame."
      );

    py::class_<PyToolFrameSpec>(
      m, "ToolFrameSpec", "Optional tool mass/TCP description for building the robot KDL chain."
    )
      .def(py::init<>())
      .def_readwrite(
        "tool_body", &PyToolFrameSpec::tool_body, "Root body of tool subtree to lump into dynamics."
      )
      .def_readwrite(
        "tcp_site", &PyToolFrameSpec::tcp_site, "MuJoCo site used as the terminal TCP frame."
      )
      .def_readwrite(
        "ft_sensors",
        &PyToolFrameSpec::ft_sensors,
        "Logical force-torque sensors attached to this robot."
      );

    py::class_<mj_kdl::ResetOptions>(m, "ResetOptions")
      .def(py::init<>())
      .def_readwrite("keyframe", &mj_kdl::ResetOptions::keyframe)
      .def_readwrite("use_keyframe", &mj_kdl::ResetOptions::use_keyframe);

    py::class_<mj_kdl::ResetInfo>(m, "ResetInfo")
      .def_readonly("used_keyframe", &mj_kdl::ResetInfo::used_keyframe)
      .def_readonly("keyframe", &mj_kdl::ResetInfo::keyframe);

    py::class_<mj_kdl::ResetContext>(
      m, "ResetContext", "Context passed to Env.on_reset after MuJoCo data is reset."
    )
      .def_property_readonly(
        "options",
        [](const mj_kdl::ResetContext &c) {
            return c.options ? *c.options : mj_kdl::ResetOptions{};
        },
        "Reset options that triggered this reset."
      )
      .def_property_readonly(
        "info",
        [](const mj_kdl::ResetContext &c) { return c.info ? *c.info : mj_kdl::ResetInfo{}; },
        "Reset result (keyframe used, etc.)."
      );

    py::class_<PyRobot, std::shared_ptr<PyRobot>>(
      m, "Robot", "Robot registered with an Env and backed by a wrapper-built KDL chain."
    )
      .def(
        "set_joint_pos",
        [](PyRobot &self, const py::object &q) {
            self.set_joint_pos(to_jnt_array(q, self.robot.n_joints));
        },
        py::arg("q"),
        "Write MuJoCo joint positions; frames read afterwards follow them."
      )
      .def("kdl_chain", &PyRobot::kdl_chain, "Return the wrapper-built chain as a PyKDL.Chain.")
      .def_property_readonly(
        "ft_sensor_names",
        &PyRobot::ft_sensor_names,
        "Configured logical force-torque sensor names."
      )
      .def(
        "ft_sensor",
        &PyRobot::ft_sensor,
        py::arg("name"),
        "Return the latest measured force-torque sensor value as a PyKDL.Wrench."
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
            return read_only_array<double>(self.robot.jnt_pos_msr);
        },
        [](PyRobot &self, const std::vector<double> &values) {
            self.set_port(&mj_kdl::RobotPorts::jnt_pos_msr, values);
        }
      )
      .def_property(
        "jnt_vel_msr",
        [](const PyRobot &self) {
            self.ensure_active();
            return read_only_array<double>(self.robot.jnt_vel_msr);
        },
        [](PyRobot &self, const std::vector<double> &values) {
            self.set_port(&mj_kdl::RobotPorts::jnt_vel_msr, values);
        }
      )
      .def_property(
        "jnt_trq_msr",
        [](const PyRobot &self) {
            self.ensure_active();
            return read_only_array<double>(self.robot.jnt_trq_msr);
        },
        [](PyRobot &self, const std::vector<double> &values) {
            self.set_port(&mj_kdl::RobotPorts::jnt_trq_msr, values);
        }
      )
      .def_property(
        "jnt_pos_cmd",
        [](const PyRobot &self) {
            self.ensure_active();
            return read_only_array<double>(self.robot.jnt_pos_cmd);
        },
        [](PyRobot &self, const std::vector<double> &values) {
            self.set_port(&mj_kdl::RobotPorts::jnt_pos_cmd, values);
        }
      )
      .def_property(
        "jnt_vel_cmd",
        [](const PyRobot &self) {
            self.ensure_active();
            return read_only_array<double>(self.robot.jnt_vel_cmd);
        },
        [](PyRobot &self, const std::vector<double> &values) {
            self.set_port(&mj_kdl::RobotPorts::jnt_vel_cmd, values);
        }
      )
      .def_property(
        "jnt_trq_cmd",
        [](const PyRobot &self) {
            self.ensure_active();
            return read_only_array<double>(self.robot.jnt_trq_cmd);
        },
        [](PyRobot &self, const std::vector<double> &values) {
            self.set_port(&mj_kdl::RobotPorts::jnt_trq_cmd, values);
        }
      )
      .def_property_readonly(
        "jnt_saturated",
        [](const PyRobot &self) {
            self.ensure_active();
            return read_only_array<bool>(self.robot.jnt_saturated);
        }
      )
      .def(
        "set_control_mode",
        [](PyRobot &self, CtrlMode mode) {
            self.ensure_active();
            if (mj_kdl::Status s = mj_kdl::set_control_mode(&self.robot, mode); !s)
                throw std::runtime_error(s.error);
        },
        py::arg("mode"),
        "Switch mode without a jump: seeds the new mode's commands from the current state."
      )
      .def_property_readonly(
        "has_tcp_frame",
        [](const PyRobot &self) {
            self.ensure_active();
            return self.robot.has_tcp_frame;
        },
        "Whether an authored TCP site terminates the KDL chain."
      )
      .def_property_readonly(
        "tcp_site",
        [](const PyRobot &self) {
            self.ensure_active();
            return self.robot.tcp_site;
        },
        "Name of the TCP site, or empty if none."
      )
      .def(
        "joint_force_limits",
        [](const PyRobot &self, double fallback) {
            self.ensure_active();
            return read_only_array<double>(mj_kdl::joint_force_limits(&self.robot, fallback));
        },
        py::arg("fallback") = 1e6,
        "Per-joint force/torque limit of the active mode's actuators; fallback where unlimited."
      )
      .def_property_readonly(
        "tip_T_tcp",
        [](const PyRobot &self) {
            self.ensure_active();
            return kdl_frame_to_py(self.robot.tip_T_tcp);
        },
        "Transform from the KDL tip frame to the TCP frame as a PyKDL.Frame."
      );

    py::class_<PyViewer>(
      m, "Viewer", "The simulate UI of an Env, opened by Env.open_viewer(); inert while closed."
    )
      .def("is_running", &PyViewer::is_running, "True while the viewer window is open.")
      .def(
        "key_pressed",
        &PyViewer::key_pressed,
        py::arg("key"),
        "True while the GLFW key code is held; False headless."
      )
      .def(
        "capture_key",
        &PyViewer::capture_key,
        py::arg("key"),
        py::arg("capture") = true,
        "Claim a GLFW key so the UI does not act on it; capture=False gives it back."
      )
      .def("clear_trace", &PyViewer::clear_trace, "Clear viewer trace geometry.")
      .def(
        "add_trace_segment",
        [](
          PyViewer                    &self,
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
        &PyViewer::use_camera,
        py::arg("name") = "",
        "Switch to a named camera; empty name restores default."
      )
      .def(
        "set_free_camera",
        &PyViewer::set_free_camera,
        py::arg("distance"),
        py::arg("azimuth"),
        py::arg("elevation"),
        py::arg("lookat") = std::array<double, 3>{ 0.0, 0.0, 0.0 },
        "Configure the free orbit camera (distance, azimuth/elevation in degrees, "
        "lookat xyz in meters)."
      )
      .def_property(
        "realtime_factor",
        [](const PyViewer &self) { return self.viewer()->realtime_factor; },
        [](PyViewer &self, double value) {
            if (value < 0.0) throw std::invalid_argument("realtime_factor must be >= 0");
            self.viewer()->realtime_factor = value;
        }
      );

    py::class_<PyVideoRecorder, std::shared_ptr<PyVideoRecorder>>(
      m, "VideoRecorder", "Offscreen MuJoCo video recorder for an Env."
    )
      .def_static(
        "open",
        [](const std::shared_ptr<PyEnv> &env, const std::string &out, int w, int h, int fps) {
            return PyVideoRecorder::make(
              env, out, w, h, mj_kdl::VideoResolution::R720p, false, fps
            );
        },
        py::arg("env"),
        py::arg("out_path"),
        py::arg("width")  = 1280,
        py::arg("height") = 720,
        py::arg("fps")    = 60,
        "Open an offscreen recorder for an Env with explicit dimensions."
      )
      .def_static(
        "open_preset",
        [](
          const std::shared_ptr<PyEnv> &env,
          const std::string            &out,
          mj_kdl::VideoResolution       res,
          int                           fps
        ) { return PyVideoRecorder::make(env, out, 0, 0, res, true, fps); },
        py::arg("env"),
        py::arg("out_path"),
        py::arg("resolution") = mj_kdl::VideoResolution::R720p,
        py::arg("fps")        = 60,
        "Open an offscreen recorder for an Env with a resolution preset."
      )
      .def_static(
        "open_offscreen",
        &PyVideoRecorder::make_offscreen,
        py::arg("env"),
        py::arg("width"),
        py::arg("height"),
        "Open an offscreen renderer (no video file) for render_rgb()."
      )
      .def(
        "render_rgb",
        &PyVideoRecorder::render_rgb,
        "Render the Env's current state as a (height, width, 3) uint8 array, top row first."
      )
      .def("__enter__", [](const std::shared_ptr<PyVideoRecorder> &self) { return self; })
      .def("__exit__", [](PyVideoRecorder &self, const py::args &) { self.close(); })
      .def(
        "record_frame",
        &PyVideoRecorder::record_frame,
        py::call_guard<py::gil_scoped_release>(),
        "Render and append one frame."
      )
      .def(
        "use_camera",
        &PyVideoRecorder::use_camera,
        py::arg("name") = "",
        "Switch to a named camera; empty name restores default."
      )
      .def(
        "set_free_camera",
        &PyVideoRecorder::set_free_camera,
        py::arg("distance"),
        py::arg("azimuth"),
        py::arg("elevation"),
        py::arg("lookat") = std::array<double, 3>{ 0.0, 0.0, 0.0 },
        "Configure the free orbit camera (distance, azimuth/elevation in degrees, "
        "lookat point). Call between frames to orbit."
      )
      .def("close", &PyVideoRecorder::close, "Finalize and close the recorder.");

    py::class_<PyEnv, std::shared_ptr<PyEnv>>(
      m, "Env", "The simulation: compiled scene, its robots, scene slots and viewer."
    )
      .def_static("build", &PyEnv::build, py::arg("spec"), "Build an environment from a SceneSpec.")
      .def("close", &PyEnv::close, "Close the viewer and free the model; robots become closed.")
      .def_property_readonly(
        "model",
        [](const PyEnv &self) {
            self.ensure_open();
            return self.model_obj;
        },
        "The live mujoco.MjModel; a new object after add_object()/remove_object()."
      )
      .def_property_readonly(
        "data",
        [](const PyEnv &self) {
            self.ensure_open();
            return self.data_obj;
        },
        "The live mujoco.MjData; a new object after add_object()/remove_object(). The viewer "
        "thread reads it too: call mujoco functions on it only while the viewer is closed or "
        "paused."
      )
      .def("__enter__", [](const std::shared_ptr<PyEnv> &self) { return self; })
      .def("__exit__", [](PyEnv &self, const py::args &) { self.close(); })
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
        "Create a robot and register it with this environment. prefix is prepended to every "
        "name it resolves: bodies, tool body, TCP site, F/T sensors."
      )
      .def(
        "create_robot_from_chain",
        [](
          const std::shared_ptr<PyEnv>   &self,
          const py::object               &chain,
          const std::vector<std::string> &joint_names,
          const std::string              &prefix,
          const py::object               &tool
        ) {
            const auto cpp_chain = chain.cast<KDL::Chain>();
            if (tool.is_none())
                return self->create_robot_from_chain(cpp_chain, joint_names, prefix, nullptr);
            auto cpp_tool = tool.cast<PyToolFrameSpec>();
            return self->create_robot_from_chain(cpp_chain, joint_names, prefix, &cpp_tool);
        },
        py::arg("chain"),
        py::arg("joint_names"),
        py::arg("prefix") = "",
        py::arg("tool")   = py::none(),
        "Register a robot driven by a given PyKDL.Chain; joint_names are the MuJoCo joints in "
        "chain order. The chain is used as is (no tool inertia lumped); tool only names FT "
        "sensors."
      )
      .def(
        "step",
        &PyEnv::step,
        py::call_guard<py::gil_scoped_release>(),
        "Advance one timestep; with the viewer open, honours its pause and perturbation. "
        "Returns False once the viewer window is closed."
      )
      .def(
        "update",
        &PyEnv::update,
        "One control cycle: read every robot and scene slot, then apply their commands."
      )
      .def(
        "pace",
        &PyEnv::pace,
        py::call_guard<py::gil_scoped_release>(),
        "Sleep out this step's share of wall time at the viewer's real-time factor; no-op "
        "headless. step() never sleeps."
      )
      .def(
        "open_viewer",
        &PyEnv::open_viewer,
        py::arg("title") = "MuJoCo",
        "Open the simulate UI; step() drives it. Closed by close()."
      )
      .def_property_readonly(
        "viewer",
        [](const std::shared_ptr<PyEnv> &self) { return PyViewer{ self }; },
        "The simulate UI; inert until open_viewer()."
      )
      .def(
        "reset",
        [](PyEnv &self, const py::object &options) {
            std::optional<mj_kdl::ResetOptions> cpp_options;
            if (!options.is_none()) cpp_options = options.cast<mj_kdl::ResetOptions>();
            py::gil_scoped_release nogil;
            return self.reset(cpp_options ? &*cpp_options : nullptr);
        },
        py::arg("options") = py::none(),
        "Reset MuJoCo state, re-seed every robot and scene slot, then call on_reset; what it "
        "moves is read back."
      )
      .def(
        "add_object",
        &PyEnv::add_object,
        py::arg("object"),
        py::call_guard<py::gil_scoped_release>(),
        "Rebuild with an added object; robots and scene slots follow the new model."
      )
      .def(
        "set_control_mode",
        &PyEnv::set_control_mode,
        py::arg("robot"),
        py::arg("mode"),
        "Switch robot (its SceneSpec.robots index) to mode, for robots driven without a Robot."
      )
      .def(
        "remove_object",
        &PyEnv::remove_object,
        py::arg("name"),
        py::call_guard<py::gil_scoped_release>(),
        "Rebuild without the named object; robots and scene slots follow the new model."
      )
      .def_property(
        "on_reset",
        [](const PyEnv &self) { return self.reset_callback; },
        [](PyEnv &self, const py::object &cb) {
            if (!cb.is_none() && !PyCallable_Check(cb.ptr())) {
                throw std::invalid_argument("on_reset must be callable or None");
            }
            self.reset_callback = cb;
        },
        "Callable invoked by reset() after every robot and slot is re-seeded; what it moves is "
        "read back. Receives a ResetContext. Set to None to clear."
      )
      .def(
        "body_frame",
        &PyEnv::body_frame,
        py::arg("name"),
        "Return a body world pose as PyKDL.Frame."
      )
      .def(
        "site_frame",
        &PyEnv::site_frame,
        py::arg("name"),
        "Return a site world pose as PyKDL.Frame."
      )
      .def(
        "set_body_pose",
        [](
          PyEnv                       &self,
          const std::string           &name,
          const std::array<double, 3> &pos,
          const py::object            &quat
        ) {
            if (quat.is_none()) return self.set_body_pose(name, pos, nullptr);
            auto q = quat.cast<std::array<double, 4>>();
            return self.set_body_pose(name, pos, &q);
        },
        py::arg("name"),
        py::arg("pos"),
        py::arg("quat") = py::none(),
        "Set a free body pose (position and optional xyzw quaternion)."
      )
      .def(
        "save_model_xml",
        &PyEnv::save_model_xml,
        py::arg("path"),
        "Save the compiled model to an MJCF XML file."
      )
      .def_readonly("spec", &PyEnv::spec, "Python scene spec used to build this environment.");

    m.def("set_log_level", &mj_kdl::set_log_level, py::arg("level"), "Set wrapper log level.");
    m.def("get_log_level", &mj_kdl::get_log_level, "Return wrapper log level.");
}
