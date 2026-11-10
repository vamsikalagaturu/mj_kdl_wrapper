#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#ifdef MJ_KDL_PYTHON_RENDER
#include "mj_kdl_wrapper/render.hpp"
#endif
#ifdef MJ_KDL_PYTHON_VIEWER
#include "mj_kdl_wrapper/viewer.hpp"
#endif

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
    std::vector<mj_kdl::CtrlModeSpec>                modes;
    std::vector<std::pair<std::string, std::string>> contact_exclusions;
};

struct PyRobotSpec
{
    std::string                       path;
    std::string                       prefix;
    PyAttachTarget                    attach_to;
    std::array<double, 3>             pos  = { 0.0, 0.0, 0.0 };
    std::array<double, 4>             quat = { 0.0, 0.0, 0.0, 1.0 };
    std::vector<PyAttachmentSpec>     attachments;
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
    std::optional<bool>        use_stage;
    std::string                stage;
    double                     floor_z = 0.0;
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
    out.modes              = src.modes;
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
        if (!src.mass && !src.fixed)
            throw std::runtime_error("SceneObject.mass must be set for non-fixed primitives");
        if (!src.friction) {
            throw std::runtime_error("SceneObject.friction must be set for primitive objects");
        }
        std::copy(src.size->begin(), src.size->end(), out.size);
        if (src.mass) out.mass = *src.mass;
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
    if (!src.use_stage) throw std::runtime_error("SceneSpec.use_stage must be set");
    out.timestep  = *src.timestep;
    out.gravity_z = src.gravity_z;
    out.use_stage = *src.use_stage;
    out.stage     = src.stage;
    out.floor_z   = src.floor_z;
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
    if (py::isinstance<KDL::JntArray>(values)) {
        const auto &q = values.cast<const KDL::JntArray &>();
        if (static_cast<int>(q.rows()) != expected) {
            throw std::invalid_argument(
              "expected " + std::to_string(expected) + " joint values, got "
              + std::to_string(q.rows())
            );
        }
        return q;
    }
    return to_jnt_array(values.cast<std::vector<double>>(), expected);
}

// A copy, not a view: reset() replaces the port buffers a view would point into.
template<typename T, typename Values> py::array_t<T> read_only_array(const Values &values)
{
    py::array_t<T> out(static_cast<py::ssize_t>(values.size()));
    std::copy(values.begin(), values.end(), out.mutable_data());
    py::detail::array_proxy(out.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
    return out;
}

// The KDL types are registered by PyKDL, so a cast needs it imported once.
void import_pykdl()
{
    static const bool imported = (py::module_::import("PyKDL"), true);
    (void)imported;
}

template<typename T> py::object to_pykdl(const T &value)
{
    import_pykdl();
    try {
        return py::cast(value);
    } catch (const py::cast_error &err) {
        throw std::runtime_error(
          std::string("PyKDL did not register a compatible KDL caster: ") + err.what()
        );
    }
}

bool holder_constructed(PyObject *self)
{
    return reinterpret_cast<py::detail::instance *>(self)
      ->get_value_and_holder()
      .holder_constructed();
}

// Shows the cycle collector the Python object T keeps: an Env's callback, a handle's Env.
template<typename T, py::object T::*Ref> int gc_traverse(PyObject *self, visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));
    if (holder_constructed(self)) Py_VISIT((py::cast<T &>(py::handle(self)).*Ref).ptr());
    return 0;
}

template<typename T, py::object T::*Ref> int gc_clear(PyObject *self)
{
    if (holder_constructed(self)) py::cast<T &>(py::handle(self)).*Ref = py::object();
    return 0;
}

template<typename T, py::object T::*Ref> py::custom_type_setup gc_keeps(bool clears)
{
    return py::custom_type_setup([clears](PyHeapTypeObject *heap) {
        heap->ht_type.tp_flags |= Py_TPFLAGS_HAVE_GC;
        heap->ht_type.tp_traverse = gc_traverse<T, Ref>;
        heap->ht_type.tp_clear    = clears ? gc_clear<T, Ref> : nullptr;
    });
}

struct PyResetContext
{
    mj_kdl::ResetOptions options;
    mj_kdl::ResetInfo    info;
};

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
    std::optional<py::error_already_set> reset_error; // raised once the C++ reset has finished

    ~PyEnv() { close(); }

    void ensure_open() const
    {
        if (!env.model || !env.data) throw std::runtime_error("env is closed");
    }

    void wire_reset_hook()
    {
        env.on_reset = [this](mj_kdl::ResetContext *ctx) {
            py::gil_scoped_acquire gil;
            if (!reset_callback || reset_callback.is_none()) return;
            try {
                reset_callback(PyResetContext{ *ctx->options, *ctx->info });
            } catch (py::error_already_set &err) {
                if (!reset_error) reset_error = std::move(err);
            }
        };
    }

    void raise_reset_error()
    {
        if (!reset_error) return;
        py::error_already_set err = std::move(*reset_error);
        reset_error.reset();
        throw err;
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

    // cleanup frees nothing of the adopted pair, so the mujoco objects are dropped after it.
    void close()
    {
        {
            py::gil_scoped_release nogil;
            mj_kdl::cleanup(&env);
        }
        model_obj      = py::object();
        data_obj       = py::object();
        reset_callback = py::object();
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
        mj_kdl::ResetInfo info;
        {
            py::gil_scoped_release nogil;
            info = mj_kdl::reset(&env, options);
        }
        raise_reset_error();
        return info;
    }

    // An EnvHooks::on_step may reset the Env, so on_reset can fail here too; it may also rebuild
    // it (the viewer's Reload), and then the new mujoco objects are taken here.
    bool step()
    {
        ensure_open();
        bool running = false;
        {
            py::gil_scoped_release nogil;
            running = mj_kdl::step(&env);
        }
        if (next_model) take_next();
        raise_reset_error();
        return running;
    }

    void update()
    {
        ensure_open();
        py::gil_scoped_release nogil;
        mj_kdl::update(&env);
    }

    void add_trace_segment(
      const std::array<double, 3> &a,
      const std::array<double, 3> &b,
      const std::array<float, 4>  *rgba
    )
    {
        const KDL::Vector ka(a[0], a[1], a[2]);
        const KDL::Vector kb(b[0], b[1], b[2]);
        mj_kdl::add_trace_segment(&env, ka, kb, rgba ? rgba->data() : nullptr);
    }

    void add_overlay_arrow(
      const std::array<double, 3> &from,
      const std::array<double, 3> &dir,
      double                       length,
      const std::array<float, 4>  *rgba
    )
    {
        const KDL::Vector kfrom(from[0], from[1], from[2]);
        const KDL::Vector kdir(dir[0], dir[1], dir[2]);
        mj_kdl::add_overlay_arrow(&env, kfrom, kdir, length, rgba ? rgba->data() : nullptr);
    }

    // mujoco.MjvGeom exposes no address, so each one is made the way mujoco makes geoms.
    py::list overlay_geoms() const
    {
        py::module_ mujoco = py::module_::import("mujoco");
        py::list    out;
        for (const mjvGeom &g : mj_kdl::overlay_geoms(&env)) {
            py::object geom = mujoco.attr("MjvGeom")();
            mujoco.attr("mjv_initGeom")(
              geom,
              g.type,
              py::array_t<mjtNum>(
                3, std::array<mjtNum, 3>{ g.size[0], g.size[1], g.size[2] }.data()
              ),
              py::array_t<mjtNum>(3, std::array<mjtNum, 3>{ g.pos[0], g.pos[1], g.pos[2] }.data()),
              py::array_t<mjtNum>(
                9,
                std::array<mjtNum, 9>{ g.mat[0],
                                       g.mat[1],
                                       g.mat[2],
                                       g.mat[3],
                                       g.mat[4],
                                       g.mat[5],
                                       g.mat[6],
                                       g.mat[7],
                                       g.mat[8] }
                  .data()
              ),
              py::array_t<float>(4, g.rgba)
            );
            out.append(geom);
        }
        return out;
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
        if (mj_kdl::Status s = mj_kdl::scene_remove_object(&env, name); !s)
            throw std::runtime_error(s.error);
        take_next();
        auto it = std::find_if(spec.objects.begin(), spec.objects.end(), [&](const auto &obj) {
            return obj.name == name;
        });
        if (it != spec.objects.end()) spec.objects.erase(it);
    }

    void set_control_mode(int robot, CtrlMode mode)
    {
        ensure_open();
        mj_kdl::Status s;
        {
            py::gil_scoped_release nogil;
            s = mj_kdl::set_control_mode(&env, robot, mode);
        }
        if (!s) throw std::runtime_error(s.error);
    }

    py::object body_frame(const std::string &name)
    {
        ensure_open();
        KDL::Frame frame;
        bool       found = false;
        {
            py::gil_scoped_release nogil;
            found = mj_kdl::get_body_frame(&env, name.c_str(), &frame);
        }
        if (!found) throw std::runtime_error("body not found");
        return to_pykdl(frame);
    }

    py::object site_frame(const std::string &name)
    {
        ensure_open();
        KDL::Frame frame;
        bool       found = false;
        {
            py::gil_scoped_release nogil;
            found = mj_kdl::get_site_frame(&env, name.c_str(), &frame);
        }
        if (!found) throw std::runtime_error("site not found");
        return to_pykdl(frame);
    }

    void set_body_pose(
      const std::string           &name,
      const std::array<double, 3> &pos,
      const std::array<double, 4> *quat_xyzw
    )
    {
        ensure_open();
        py::gil_scoped_release nogil;
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
    // The Python Env, declared before robot so the robot unregisters while the Env still exists.
    py::object    env_owner;
    mj_kdl::Robot robot;

    void ensure_active() const
    {
        if (!robot.model || !robot.data || robot.n_joints <= 0) {
            throw std::runtime_error("robot is closed");
        }
    }

    std::vector<std::pair<double, double>> joint_limits() const
    {
        ensure_active();
        return robot.joint_limits;
    }

    // PyKDL solvers keep a reference to their chain without keeping it alive, so no temporary.
    py::object kdl_chain(py::object self)
    {
        ensure_active();
        import_pykdl();
        return py::cast(&robot.chain, py::return_value_policy::reference_internal, self);
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
            if (sensor.name == name) return to_pykdl(sensor.wrench);
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
    out->env_owner = py::cast(shared_from_this());
    mj_kdl::ToolFrameSpec cpp_tool;
    if (tool) cpp_tool = to_cpp(*tool);
    mj_kdl::Status s;
    {
        py::gil_scoped_release nogil;
        s = mj_kdl::init_robot_from_mjcf(
          &out->robot,
          &env,
          base_body.c_str(),
          tip_body.c_str(),
          prefix.c_str(),
          tool ? &cpp_tool : nullptr
        );
    }
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
    out->env_owner = py::cast(shared_from_this());
    mj_kdl::ToolFrameSpec cpp_tool;
    if (tool) cpp_tool = to_cpp(*tool);
    mj_kdl::Status s;
    {
        py::gil_scoped_release nogil;
        s = mj_kdl::init_robot_from_chain(
          &out->robot, &env, chain, joint_names, prefix.c_str(), tool ? &cpp_tool : nullptr
        );
    }
    if (!s) throw std::runtime_error(s.error);
    return out;
}

#ifdef MJ_KDL_PYTHON_VIEWER
// The C++ viewer on an Env; while it is closed, pace() does nothing and the queries are False.
// Calls drop the GIL: the window's thread never takes it, the Env's thread may wait on the window.
struct PyViewer
{
    mj_kdl::Viewer viewer;
    py::object     owner; // the Python Env, which keeps the Env alive while the window is open
    PyEnv         *env = nullptr;

    void open(const std::shared_ptr<PyEnv> &target, const std::string &title)
    {
        target->ensure_open();
        mj_kdl::Status s;
        {
            py::gil_scoped_release nogil;
            s = mj_kdl::open_viewer(&viewer, &target->env, title.c_str());
        }
        if (!s) throw std::runtime_error(s.error);
        owner = py::cast(target);
        env   = target.get();
    }

    void close()
    {
        {
            py::gil_scoped_release nogil;
            mj_kdl::cleanup(&viewer);
        }
        owner = py::object();
        env   = nullptr;
    }

    bool use_camera(const std::string &name)
    {
        if (!env) throw std::runtime_error("viewer is not open");
        env->ensure_open();
        py::gil_scoped_release nogil;
        return mj_kdl::use_camera(&viewer, env->env.model, name.empty() ? nullptr : name.c_str());
    }
};
#endif

#ifdef MJ_KDL_PYTHON_RENDER
// The C++ recorder on an Env: an MP4 through ffmpeg, or offscreen frames for render_rgb().
struct PyVideoRecorder
{
    mj_kdl::VideoRecorder recorder;
    py::object            owner; // the Python Env, which keeps the Env alive
    PyEnv                *env    = nullptr;
    bool                  active = false;
    int                   width  = 0;
    int                   height = 0;

    ~PyVideoRecorder() { close(); }

    // out_path empty: offscreen only, no video file.
    static std::shared_ptr<PyVideoRecorder> make(
      const std::shared_ptr<PyEnv> &target,
      const std::string            &out_path,
      int                           width,
      int                           height,
      int                           fps
    )
    {
        if (!target) throw std::runtime_error("env is null");
        target->ensure_open();
        mjModel *m   = target->env.model;
        auto     out = std::shared_ptr<PyVideoRecorder>(new PyVideoRecorder());
        out->owner   = py::cast(target);
        out->env     = target.get();
        const mj_kdl::Status s =
          out_path.empty()
            ? mj_kdl::init_offscreen(&out->recorder, m, width, height)
            : mj_kdl::init_video_recorder(&out->recorder, m, out_path.c_str(), width, height, fps);
        if (!s) throw std::runtime_error(s.error);
        out->active = true;
        out->width  = width;
        out->height = height;
        return out;
    }

    void ensure_open() const
    {
        if (!active || !env->env.model) throw std::runtime_error("recorder is closed");
    }

    py::array_t<std::uint8_t> render_rgb()
    {
        ensure_open();
        py::array_t<std::uint8_t> out({ height, width, 3 });
        std::uint8_t             *pixels = out.mutable_data();
        bool                      ok     = false;
        {
            py::gil_scoped_release nogil;
            ok = mj_kdl::render_rgb(&recorder, &env->env, pixels);
        }
        if (!ok) throw std::runtime_error("render_rgb failed");
        return out;
    }

    bool record_frame()
    {
        ensure_open();
        py::gil_scoped_release nogil;
        return mj_kdl::record_frame(&recorder, &env->env);
    }

    bool use_camera(const std::string &name)
    {
        ensure_open();
        if (name.empty()) {
            mjv_defaultFreeCamera(env->env.model, &recorder.cam);
            return true;
        }
        const int id = mj_name2id(env->env.model, mjOBJ_CAMERA, name.c_str());
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
        {
            py::gil_scoped_release nogil;
            mj_kdl::cleanup(&recorder);
        }
        active = false;
        owner  = py::object();
        env    = nullptr;
    }
};
#endif

} // namespace

PYBIND11_MODULE(_mj_kdl_wrapper, m)
{
    m.doc()                      = "Python bindings for mj_kdl_wrapper";
    m.attr("__version__")        = MJ_KDL_WRAPPER_VERSION;
    m.attr("__mujoco_version__") = mj_versionString();

    py::enum_<mj_kdl::LogLevel>(
      m, "LogLevel", "Log threshold: messages at or above it print; NONE prints nothing."
    )
      .value("INFO", mj_kdl::LogLevel::INFO)
      .value("WARN", mj_kdl::LogLevel::WARN)
      .value("ERROR", mj_kdl::LogLevel::ERROR)
      .value("NONE", mj_kdl::LogLevel::NONE);

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

    py::class_<PyAttachTarget>(
      m,
      "AttachTarget",
      "Placement target in an accumulated scene spec. World is the default; other kinds use name."
    )
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
        "modes", &PyAttachmentSpec::modes, "Control modes for its own joints; [] = native only."
      )
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
      m, "SceneSpec", "Full scene description. timestep and use_stage are required."
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
        "use_stage",
        &PySceneSpec::use_stage,
        "Required: build on a stage (floor, sky, lights), or on an empty world."
      )
      .def_readwrite(
        "stage", &PySceneSpec::stage, "Stage MJCF path; empty uses the bundled default stage."
      )
      .def_readwrite(
        "floor_z", &PySceneSpec::floor_z, "Added to the height of the stage's planes [m]."
      )
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

    py::class_<PyResetContext>(
      m, "ResetContext", "A copy of the reset's options and result, passed to Env.on_reset."
    )
      .def_readonly("options", &PyResetContext::options, "Reset options that triggered this reset.")
      .def_readonly("info", &PyResetContext::info, "Reset result (keyframe used, etc.).");

    py::class_<PyRobot, std::shared_ptr<PyRobot>>(
      m,
      "Robot",
      "Robot registered with an Env and backed by a wrapper-built KDL chain.",
      gc_keeps<PyRobot, &PyRobot::env_owner>(false)
    )
      .def(
        "set_joint_pos",
        [](PyRobot &self, const py::object &q) {
            self.ensure_active();
            const KDL::JntArray    joints = to_jnt_array(q, self.robot.n_joints);
            py::gil_scoped_release nogil;
            mj_kdl::set_joint_pos(&self.robot, joints);
        },
        py::arg("q"),
        "Write MuJoCo joint positions; frames read afterwards follow them."
      )
      .def(
        "kdl_chain",
        [](py::object self) { return self.cast<PyRobot &>().kdl_chain(self); },
        "The robot's own chain as a PyKDL.Chain, valid while the robot lives."
      )
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
            mj_kdl::Status s;
            {
                py::gil_scoped_release nogil;
                s = mj_kdl::set_control_mode(&self.robot, mode);
            }
            if (!s) throw std::runtime_error(s.error);
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
            return to_pykdl(self.robot.tip_T_tcp);
        },
        "Transform from the KDL tip frame to the TCP frame as a PyKDL.Frame."
      );

    py::class_<PyEnv, std::shared_ptr<PyEnv>>(
      m,
      "Env",
      "The simulation: compiled scene and its robots. View it with Viewer, record or render it "
      "with VideoRecorder.",
      gc_keeps<PyEnv, &PyEnv::reset_callback>(true)
    )
      .def_static("build", &PyEnv::build, py::arg("spec"), "Build an environment from a SceneSpec.")
      .def("close", &PyEnv::close, "Free the model; robots become closed.")
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
        "The live mujoco.MjData; a new object after add_object()/remove_object()."
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
      .def("step", &PyEnv::step, "Advance one timestep. step() never sleeps.")
      .def(
        "update", &PyEnv::update, "One control cycle: read every robot, then apply its commands."
      )
      .def(
        "clear_trace",
        [](PyEnv &self) { mj_kdl::clear_trace(&self.env); },
        "Remove every overlay geom: segments and arrows."
      )
      .def(
        "add_trace_segment",
        [](
          PyEnv                       &self,
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
        "Add an overlay line segment in world coordinates [m]."
      )
      .def(
        "add_overlay_arrow",
        [](
          PyEnv                       &self,
          const std::array<double, 3> &from,
          const std::array<double, 3> &dir,
          double                       length,
          const py::object            &rgba
        ) {
            if (rgba.is_none()) {
                self.add_overlay_arrow(from, dir, length, nullptr);
                return;
            }
            auto color = rgba.cast<std::array<float, 4>>();
            self.add_overlay_arrow(from, dir, length, &color);
        },
        py::arg("from_"),
        py::arg("dir"),
        py::arg("length"),
        py::arg("rgba") = py::none(),
        "Add an overlay arrow from a point along dir, length [m] long, in world coordinates."
      )
      .def(
        "overlay_geoms",
        &PyEnv::overlay_geoms,
        "The overlay geoms as mujoco.MjvGeom; Viewer and VideoRecorder draw them themselves."
      )
      .def(
        "reset",
        [](PyEnv &self, const py::object &options) {
            std::optional<mj_kdl::ResetOptions> cpp_options;
            if (!options.is_none()) cpp_options = options.cast<mj_kdl::ResetOptions>();
            return self.reset(cpp_options ? &*cpp_options : nullptr);
        },
        py::arg("options") = py::none(),
        "Reset MuJoCo state, re-seed every robot, then call on_reset; what it moves is read back."
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
        [](const PyEnv &self) {
            return self.reset_callback ? self.reset_callback : py::object(py::none());
        },
        [](PyEnv &self, const py::object &cb) {
            if (!cb.is_none() && !PyCallable_Check(cb.ptr())) {
                throw std::invalid_argument("on_reset must be callable or None");
            }
            self.reset_callback = cb;
        },
        "Callable invoked by reset() after every robot is re-seeded; what it moves is read back. "
        "Receives a ResetContext; an exception it raises comes out of reset() or step()."
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
      .def_property_readonly(
        "spec",
        [](const PyEnv &self) { return self.spec; },
        "A copy of the SceneSpec the Env runs, with objects added or removed since build()."
      );

#ifdef MJ_KDL_PYTHON_VIEWER
    py::class_<PyViewer>(
      m,
      "Viewer",
      "The C++ MuJoCo Studio viewer on an Env: open() it, then env.step() and pace() as in C++. "
      "Its pause, reset, perturbation and edits reach the Env inside env.step(), which returns "
      "False once the window is closed.",
      gc_keeps<PyViewer, &PyViewer::owner>(false)
    )
      .def(py::init<>())
      .def(
        "open",
        &PyViewer::open,
        py::arg("env"),
        py::arg("title") = "MuJoCo",
        "Open the window on env; raises when there is no display."
      )
      .def("close", &PyViewer::close, "Close the window.")
      .def("__enter__", [](PyViewer &self) -> PyViewer & { return self; })
      .def("__exit__", [](PyViewer &self, const py::args &) { self.close(); })
      .def(
        "is_running",
        [](const PyViewer &self) { return mj_kdl::is_running(&self.viewer); },
        "True while the window is open."
      )
      .def(
        "pace",
        [](PyViewer &self) {
            py::gil_scoped_release nogil;
            mj_kdl::pace_realtime(&self.viewer);
        },
        "Sleep out one timestep at the viewer's real-time factor; nothing while it is closed."
      )
      .def(
        "key_pressed",
        [](const PyViewer &self, int key) { return mj_kdl::key_pressed(&self.viewer, key); },
        py::arg("key"),
        "True while the key (an ImGuiKey code, as in imgui.h) is held in the window."
      )
      .def(
        "capture_key",
        [](PyViewer &self, int key, bool capture) {
            mj_kdl::capture_key(&self.viewer, key, capture);
        },
        py::arg("key"),
        py::arg("capture") = true,
        "Claim an ImGuiKey code so the viewer's own bindings leave it alone; False gives it back."
      )
      .def(
        "use_camera",
        &PyViewer::use_camera,
        py::arg("name") = "",
        "Switch to a named model camera; an empty name returns to the free camera."
      )
      .def(
        "set_free_camera",
        [](
          PyViewer                    &self,
          double                       distance,
          double                       azimuth,
          double                       elevation,
          const std::array<double, 3> &lookat
        ) { mj_kdl::set_free_camera(&self.viewer, distance, azimuth, elevation, lookat); },
        py::arg("distance"),
        py::arg("azimuth"),
        py::arg("elevation"),
        py::arg("lookat") = std::array<double, 3>{ 0.0, 0.0, 0.0 },
        "Configure the free camera: distance [m], azimuth and elevation [deg], lookat [m]."
      )
      .def_property(
        "realtime_factor",
        [](const PyViewer &self) { return self.viewer.realtime_factor; },
        [](PyViewer &self, double value) {
            if (value < 0.0) throw std::invalid_argument("realtime_factor must be >= 0");
            self.viewer.realtime_factor = value;
        },
        "pace()'s speed: 1.0 real time, 0.5 half speed, 0.0 uncapped; the window's slider sets it."
      )
      .def_property_readonly(
        "fps", [](const PyViewer &self) { return self.viewer.fps; }, "The window's frame rate."
      );
#endif

#ifdef MJ_KDL_PYTHON_RENDER
    py::enum_<mj_kdl::VideoResolution>(m, "VideoResolution")
      .value("R360p", mj_kdl::VideoResolution::R360p)
      .value("R480p", mj_kdl::VideoResolution::R480p)
      .value("R720p", mj_kdl::VideoResolution::R720p)
      .value("R1080p", mj_kdl::VideoResolution::R1080p)
      .value("R2K", mj_kdl::VideoResolution::R2K)
      .value("R4K", mj_kdl::VideoResolution::R4K);

    py::class_<PyVideoRecorder, std::shared_ptr<PyVideoRecorder>>(
      m,
      "VideoRecorder",
      "The C++ offscreen recorder (Filament) on an Env: an MP4 through ffmpeg, or RGB frames.",
      gc_keeps<PyVideoRecorder, &PyVideoRecorder::owner>(false)
    )
      .def_static(
        "open",
        [](const std::shared_ptr<PyEnv> &env, const std::string &out, int w, int h, int fps) {
            if (out.empty()) throw std::invalid_argument("out_path must be set");
            return PyVideoRecorder::make(env, out, w, h, fps);
        },
        py::arg("env"),
        py::arg("out_path"),
        py::arg("width")  = 1280,
        py::arg("height") = 720,
        py::arg("fps")    = 60,
        "Open a recorder writing an MP4 of the given size."
      )
      .def_static(
        "open_preset",
        [](
          const std::shared_ptr<PyEnv> &env,
          const std::string            &out,
          mj_kdl::VideoResolution       res,
          int                           fps
        ) {
            if (out.empty()) throw std::invalid_argument("out_path must be set");
            // The size init_video_recorder gives a preset, kept so render_rgb knows the frame.
            const int h = static_cast<int>(res);
            const int w = ((h * 16 / 9) + 1) / 2 * 2;
            return PyVideoRecorder::make(env, out, w, h, fps);
        },
        py::arg("env"),
        py::arg("out_path"),
        py::arg("resolution") = mj_kdl::VideoResolution::R720p,
        py::arg("fps")        = 60,
        "Open a recorder writing an MP4 at a 16:9 resolution preset."
      )
      .def_static(
        "open_offscreen",
        [](const std::shared_ptr<PyEnv> &env, int w, int h) {
            return PyVideoRecorder::make(env, "", w, h, 0);
        },
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
      .def("record_frame", &PyVideoRecorder::record_frame, "Render and append one frame.")
      .def(
        "use_camera",
        &PyVideoRecorder::use_camera,
        py::arg("name") = "",
        "Switch to a named model camera; an empty name returns to the free camera."
      )
      .def(
        "set_free_camera",
        &PyVideoRecorder::set_free_camera,
        py::arg("distance"),
        py::arg("azimuth"),
        py::arg("elevation"),
        py::arg("lookat") = std::array<double, 3>{ 0.0, 0.0, 0.0 },
        "Configure the free camera: distance [m], azimuth and elevation [deg], lookat [m]."
      )
      .def("__enter__", [](const std::shared_ptr<PyVideoRecorder> &self) { return self; })
      .def("__exit__", [](PyVideoRecorder &self, const py::args &) { self.close(); })
      .def("close", &PyVideoRecorder::close, "Finish the video and free the renderer.");
#endif

    m.def(
      "set_log_level",
      &mj_kdl::set_log_level,
      py::arg("level"),
      "Print wrapper messages at level and above; NONE prints nothing."
    );
    m.def("get_log_level", &mj_kdl::get_log_level, "Return the wrapper log threshold.");
}
