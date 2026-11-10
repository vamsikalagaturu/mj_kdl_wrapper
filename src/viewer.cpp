/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#include "mj_kdl_wrapper/viewer.hpp"
#include "mj_kdl_wrapper/render.hpp"

#include <SDL.h>
#include <fontconfig/fontconfig.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <implot.h>
#include <kdl/frames.hpp>
#include <mujoco/mjrfilament.h>
#include <webp/encode.h>
#include "render/filament/core/imgui_editor.h"
#include "render/filament/support/imgui_bridge.h"
#include "render/filament/support/renderer.h"
#include "studio/hal/graphics_mode.h"
#include "studio/hal/window.h"
#include "studio/render/model_scene.h"
#include "studio/sim/sim_history.h"
#include "studio/sim/sim_profiler.h"
#include "studio/sim/step_control.h"
#include "studio/ux/camera_control.h"
#include "studio/ux/console.h"
#include "studio/ux/file_dialog.h"
#include "studio/ux/fonts.h"
#include "studio/ux/gui.h"
#include "studio/ux/gui_helpers.h"
#include "studio/ux/gui_spec.h"
#include "studio/ux/imgui_widgets.h"
#include "studio/ux/interaction.h"
#include "studio/ux/picture_gui.h"
#include "studio/ux/plugin.h"
#include "studio/ux/theme.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace mj_kdl {

void    register_asset_providers();            // defined in render.cpp
mjSpec *copy_model_spec(const mjModel *model); // defined in mj_kdl_wrapper.cpp
Status  reload_scene(Env *env);                // defined in mj_kdl_wrapper.cpp
int     robot_base_body(const Robot *r);       // defined in mj_kdl_wrapper.cpp
namespace studio = mujoco::studio;
using PauseState = studio::StepControl::PauseState;

// The Recording panel's choices: every VideoResolution, by name.
static constexpr VideoResolution kRecordResolutions[] = {
    VideoResolution::R360p,  VideoResolution::R480p, VideoResolution::R720p,
    VideoResolution::R1080p, VideoResolution::R2K,   VideoResolution::R4K
};
static constexpr const char *kRecordResolutionNames[] = { "360p",  "480p", "720p",
                                                          "1080p", "2K",   "4K" };

// Recording cameras before the model's own: 0 the window's, 1 free, 2 tracking the selection.
enum RecordCamera { kRecordCurrent = 0, kRecordFree = 1, kRecordTracking = 2, kRecordFixed = 3 };

struct DataDeleter
{
    void operator()(mjData *d) const { mj_deleteData(d); }
};
using DataPtr = std::unique_ptr<mjData, DataDeleter>;

struct RecordRequest
{
    bool        start = false;
    std::string path;
    int         camera     = kRecordCurrent;
    int         resolution = 0; // index into kRecordResolutions
    int         fps        = 0;
};

// The window's actions, applied by the Env's thread inside step().
struct Commands
{
    std::optional<double>               realtime_factor;
    bool                                reset    = false;
    int                                 keyframe = -1; // reset() to it; -1: reset()'s default
    bool                                reload   = false;
    bool                                single   = false; // one step while paused
    std::optional<int>                  history_index;    // load this frame of the history
    std::optional<int>                  save_keyframe;
    std::optional<int>                  nthread;
    std::optional<std::array<float, 2>> noise;                    // control noise: scale, rate
    std::optional<std::pair<int, std::vector<mjtNum>>> state;     // State panel edit: sig, values
    std::vector<std::pair<int, mjtNum>>                ctrl;      // actuator, value
    std::vector<std::pair<int, mjtByte>>               eq_active; // equality, active
    std::optional<mjOption>                            opt;
    std::optional<RecordRequest>                       record;
    std::optional<std::string>                         save_xml;
};

// A registered robot as the Robots section shows it.
struct JointStatus
{
    std::string name;
    int         id = -1;        // MuJoCo joint id
    double      q  = 0.0;       // jnt_pos_msr [rad] or [m]
    double      lo = -INFINITY; // Robot::joint_limits; +-inf when unlimited
    double      hi = INFINITY;
};

struct RobotStatus
{
    std::string              label;  // Robot::name, else its base body, else "Robot <index>"
    std::string              prefix; // what its joint and base body names share, e.g. "r1_"
    CtrlMode                 mode     = CtrlMode::POSITION;
    bool                     paused   = false;
    int                      tcp_site = -1; // Robot::tcp_site, -1 without one
    int                      tip_body = -1; // the body its last joint moves: the TCP without a site
    std::vector<JointStatus> joints;
};

// The Env as step() last published it, for the window's thread.
struct Snapshot
{
    std::vector<RobotStatus>        robots;
    std::shared_ptr<mjModel>        model; // the window's own copy, replaced at each rebuild
    std::shared_ptr<mjSpec>         spec;  // a copy of the spec it was compiled from
    std::uint64_t                   model_gen = 0;
    DataPtr                         data;          // a full copy of the Env's mjData
    bool                            fresh = false; // data holds a copy the window has not taken
    std::vector<mjvGeom>            overlays;
    mjOption                        opt{}; // the Env's options, which the Physics panel edits
    studio::SimulationTimelineState timeline;
    double                          speed_measured = -1.0; // percent of real time
    bool                            recording      = false;
    std::string                     record_status;
};

struct ViewerImpl
{
    Viewer     *owner = nullptr;
    Env        *env   = nullptr; // nullptr once the Env has gone (EnvHooks::on_close)
    std::string title;

    std::thread             thread;
    std::mutex              mtx;
    std::condition_variable started_cv;
    bool                    started = false;
    std::string             start_error;
    std::atomic<bool>       stop{ false };   // the Env's thread asks the window to close
    std::atomic<bool>       closed{ false }; // the window is gone

    // Guarded by mtx.
    Commands                 commands;
    Snapshot                 snapshot;
    bool                     want_overlays = true; // the window has drawn the last overlays
    bool                     paused        = false;
    mjvPerturb               perturb{};
    mjvCamera                camera{};
    std::optional<mjvCamera> camera_request; // use_camera() / set_free_camera() while open
    double                   fps = 0.0;

    std::atomic<bool> keys[ImGuiKey_NamedKey_COUNT]{};
    std::atomic<bool> captured[ImGuiKey_NamedKey_COUNT]{};

    // The Env's thread only.
    std::map<int, mjtNum>                 ctrl_override; // edited controls; they win until reset
    int                                   pert_body     = -1;
    int                                   nthread       = 0;
    double                                speed_window  = 0.5; // Viewer::speed_window [s]
    float                                 noise_scale   = 0.0f;
    float                                 noise_rate    = 0.0f;
    bool                                  model_stale   = true;
    std::uint64_t                         model_gen     = 0;
    bool                                  history_stale = true;
    bool                                  stepped       = false; // the last step() integrated
    studio::SimHistory                    history;
    studio::SimulationTimelineState       timeline;
    std::chrono::steady_clock::time_point speed_wall0{};
    double                                speed_sim0     = 0.0;
    double                                speed_measured = -1.0;
    VideoRecorder                         recorder;
    bool                                  recording     = false;
    int                                   record_camera = kRecordCurrent;
    int                                   record_stride = 1;
    int                                   record_count  = 0;
};

static int key_index(int imgui_key)
{
    if (imgui_key < ImGuiKey_NamedKey_BEGIN || imgui_key >= ImGuiKey_NamedKey_END) return -1;
    return imgui_key - ImGuiKey_NamedKey_BEGIN;
}

// The Env's side

static void stop_recording(ViewerImpl *impl, const char *status)
{
    if (impl->recording) cleanup(&impl->recorder);
    impl->recording    = false;
    impl->record_count = 0;
    const std::lock_guard<std::mutex> lock(impl->mtx);
    impl->snapshot.recording     = false;
    impl->snapshot.record_status = status;
}

static void start_recording(ViewerImpl *impl, const Env *env, const RecordRequest &req)
{
    if (impl->recording) cleanup(&impl->recorder);
    const char  *path = req.path.empty() ? impl->owner->record_path.c_str() : req.path.c_str();
    const int    fps  = std::max(1, req.fps);
    const Status s    = init_video_recorder(
      &impl->recorder, env->model, path, kRecordResolutions[req.resolution], fps
    );
    impl->recording     = static_cast<bool>(s);
    impl->record_camera = req.camera;
    impl->record_count  = 0;
    impl->record_stride = std::max(1, (int)std::lround(1.0 / (fps * env->model->opt.timestep)));
    const std::lock_guard<std::mutex> lock(impl->mtx);
    impl->snapshot.recording     = impl->recording;
    impl->snapshot.record_status = s ? std::string("recording to ") + path : s.error;
}

static void record(ViewerImpl *impl, Env *env, const mjvCamera &window_cam, const mjvPerturb &pert)
{
    if (++impl->record_count < impl->record_stride) return;
    impl->record_count = 0;
    mjvCamera &cam     = impl->recorder.cam;
    cam                = window_cam;
    if (impl->record_camera == kRecordFree) {
        cam.type       = mjCAMERA_FREE;
        cam.fixedcamid = -1;
    } else if (impl->record_camera == kRecordTracking) {
        cam.type        = pert.select > 0 ? mjCAMERA_TRACKING : mjCAMERA_FREE;
        cam.trackbodyid = pert.select;
        cam.fixedcamid  = -1;
    } else if (impl->record_camera >= kRecordFixed) {
        cam.type       = mjCAMERA_FIXED;
        cam.fixedcamid = impl->record_camera - kRecordFixed;
    }
    if (!record_frame(&impl->recorder, env)) stop_recording(impl, "recording failed");
}

// Studio's StepControl::InjectNoise: Ornstein-Uhlenbeck noise on every control.
static void inject_noise(const ViewerImpl *impl, const mjModel *m, mjData *d)
{
    if (impl->noise_scale <= 0) return;
    const mjtNum rate  = mju_exp(-m->opt.timestep / impl->noise_rate);
    const mjtNum scale = impl->noise_scale * mju_sqrt(1 - rate * rate);
    for (int i = 0; i < m->nu; i++) {
        mjtNum bottom = 0, top = 0, midpoint = 0, halfrange = 1;
        if (m->actuator_ctrllimited[i]) {
            bottom    = m->actuator_ctrlrange[2 * i];
            top       = m->actuator_ctrlrange[2 * i + 1];
            midpoint  = 0.5 * (top + bottom);
            halfrange = 0.5 * (top - bottom);
        }
        d->ctrl[i] = rate * d->ctrl[i] + (1 - rate) * midpoint;
        d->ctrl[i] += scale * halfrange * mju_standardNormal(nullptr);
        if (m->actuator_ctrllimited[i]) d->ctrl[i] = mju_clip(d->ctrl[i], bottom, top);
    }
}

// After an integrated step: what Studio's App does in PostStep, and the measured speed.
static void after_step(ViewerImpl *impl, const mjModel *m, mjData *d)
{
    studio::ForEachPlugin<studio::ModelPlugin>([&](studio::ModelPlugin *plugin) {
        if (plugin->post_step) plugin->post_step(plugin, m, d);
    });
    studio::RecordHistoryFrame(impl->history, impl->timeline, m, d);

    const auto now = std::chrono::steady_clock::now();
    if (impl->speed_wall0.time_since_epoch().count() == 0 || d->time < impl->speed_sim0) {
        impl->speed_wall0 = now;
        impl->speed_sim0  = d->time;
        return;
    }
    const double wall = std::chrono::duration<double>(now - impl->speed_wall0).count();
    if (wall < impl->speed_window) return;
    impl->speed_measured = 100.0 * (d->time - impl->speed_sim0) / wall;
    impl->speed_wall0    = now;
    impl->speed_sim0     = d->time;
}

static void publish(ViewerImpl *impl, Env *env)
{
    std::shared_ptr<mjModel> model;
    std::shared_ptr<mjSpec>  spec;
    DataPtr                  data;
    if (impl->model_stale) {
        mjModel *copy = mj_copyModel(nullptr, env->model);
        if (copy) {
            model             = std::shared_ptr<mjModel>(copy, mj_deleteModel);
            data              = DataPtr(mj_makeData(copy));
            impl->model_stale = false;
            ++impl->model_gen;
            if (mjSpec *s = copy_model_spec(env->model)) spec = { s, mj_deleteSpec };
        }
    }
    const std::lock_guard<std::mutex> lock(impl->mtx);
    Snapshot                         &snap = impl->snapshot;
    if (model) {
        snap.model     = std::move(model);
        snap.spec      = std::move(spec);
        snap.data      = std::move(data);
        snap.model_gen = impl->model_gen;
        snap.fresh     = false;
    }
    // Copied only once the window took the last copy: at most one copy per window frame.
    if (!snap.fresh && snap.data) {
        mj_copyData(snap.data.get(), env->model, env->data);
        snap.fresh = true;
    }
    snap.opt            = env->model->opt;
    snap.timeline       = impl->timeline;
    snap.speed_measured = impl->speed_measured;
    // Names, ids and limits change only with the model or the robots registered.
    const bool refresh = model || snap.robots.size() != env->robots.size();
    snap.robots.resize(env->robots.size());
    for (std::size_t i = 0; i < env->robots.size(); ++i) {
        const Robot *r = env->robots[i];
        RobotStatus &s = snap.robots[i];
        if (refresh || s.joints.size() != r->joint_names.size()) {
            // Unnamed, a robot goes by its base body, e.g. "base_link" or "r2_base_link".
            const int        base = robot_base_body(r);
            const char      *body = base > 0 ? mj_id2name(env->model, mjOBJ_BODY, base) : nullptr;
            std::string_view base_name = body ? body : "";
            s.label = !base_name.empty() ? std::string(base_name) : "Robot " + std::to_string(i);
            s.joints.resize(r->joint_names.size());
            for (std::size_t j = 0; j < s.joints.size(); ++j) {
                JointStatus &js = s.joints[j];
                js.name         = r->joint_names[j];
                js.id           = mj_name2id(env->model, mjOBJ_JOINT, js.name.c_str());
                if (j < r->joint_limits.size()) std::tie(js.lo, js.hi) = r->joint_limits[j];
            }
            s.tip_body = !s.joints.empty() && s.joints.back().id >= 0
                           ? env->model->jnt_bodyid[s.joints.back().id]
                           : -1;
            // The prefix init_robot_from_mjcf() was given is not kept (it resolves full names);
            // it is what the base body and joint names share, up to their last '_'.
            for (const JointStatus &js : s.joints) {
                std::size_t n = 0;
                while (n < base_name.size() && n < js.name.size() && base_name[n] == js.name[n])
                    ++n;
                base_name = base_name.substr(0, n);
            }
            const std::size_t cut = base_name.rfind('_');
            s.prefix =
              cut == std::string_view::npos ? "" : std::string(base_name.substr(0, cut + 1));
        }
        if (!r->name.empty() && s.label != r->name) s.label = r->name;
        s.mode   = r->ctrl_mode;
        s.paused = r->paused;
        s.tcp_site =
          r->tcp_site.empty() ? -1 : mj_name2id(env->model, mjOBJ_SITE, r->tcp_site.c_str());
        for (std::size_t j = 0; j < s.joints.size(); ++j)
            s.joints[j].q = j < r->jnt_pos_msr.size() ? r->jnt_pos_msr[j] : 0.0;
    }
    if (impl->want_overlays) {
        snap.overlays       = overlay_geoms(env);
        impl->want_overlays = false;
    }
}

static bool on_step(ViewerImpl *impl, Env *env, bool *integrate)
{
    if (impl->closed) return false;
    Commands   cmd;
    bool       paused = false;
    mjvPerturb pert;
    mjvCamera  cam;
    {
        const std::lock_guard<std::mutex> lock(impl->mtx);
        cmd                 = std::move(impl->commands);
        impl->commands      = Commands{};
        paused              = impl->paused;
        pert                = impl->perturb;
        cam                 = impl->camera;
        impl->owner->fps    = impl->fps;
        impl->owner->paused = impl->paused;
    }
    if (cmd.realtime_factor) {
        impl->owner->realtime_factor = *cmd.realtime_factor;
        impl->owner->_tick_t         = {};
    }
    // The state is the one the last step() integrated to, before anything below changes it.
    if (impl->stepped) after_step(impl, env->model, env->data);
    impl->stepped = false;

    // A reload swaps the model and data the rest works on.
    if (cmd.reload) {
        if (Status s = reload_scene(env); !s) MJ_LOG_ERROR("reload failed: " << s.error);
    }
    mjModel *m = env->model;
    mjData  *d = env->data;
    if (cmd.reset) {
        ResetOptions options;
        options.keyframe = cmd.keyframe;
        reset(env, cmd.keyframe >= 0 ? &options : nullptr);
    }
    if (impl->history_stale) {
        studio::ResetHistory(impl->history, impl->timeline, m, d);
        impl->history_stale  = false;
        impl->speed_wall0    = {};
        impl->speed_measured = -1.0;
    }
    if (cmd.history_index) {
        studio::LoadHistoryFrame(impl->history, m, d, *cmd.history_index);
        impl->timeline.SetHistoryIndex(impl->history.GetIndex());
    }
    if (cmd.save_keyframe && *cmd.save_keyframe >= 0 && *cmd.save_keyframe < m->nkey) {
        mj_setKeyframe(m, d, *cmd.save_keyframe);
        impl->model_stale = true; // the window's copy holds the keyframes too
    }
    if (cmd.state) mj_setState(m, d, cmd.state->second.data(), cmd.state->first);
    if (cmd.noise) {
        impl->noise_scale = (*cmd.noise)[0];
        impl->noise_rate  = (*cmd.noise)[1];
    }
    if (cmd.nthread) impl->nthread = *cmd.nthread;
    // A no-op at the same count; a rebuilt Env has new data without a pool.
    if (cmd.nthread || impl->model_stale) mju_threadpool(d, impl->nthread);
    if (cmd.opt) {
        // Control modes own disableactuator; the panel edits everything else.
        const int modes        = m->opt.disableactuator;
        m->opt                 = *cmd.opt;
        m->opt.disableactuator = modes;
    }
    for (const auto &[eq, active] : cmd.eq_active)
        if (eq >= 0 && eq < m->neq) d->eq_active[eq] = active;
    for (const auto &[a, value] : cmd.ctrl)
        if (a >= 0 && a < m->nu) impl->ctrl_override[a] = value;
    for (const auto &[a, value] : impl->ctrl_override) d->ctrl[a] = value;
    if (cmd.save_xml) save_model_xml(m, cmd.save_xml->c_str()); // it logs either way
    if (cmd.record) {
        if (cmd.record->start)
            start_recording(impl, env, *cmd.record);
        else
            stop_recording(impl, "recording stopped");
    }

    if (!paused || cmd.single) {
        // Only the dragged body's xfrc_applied is ours: user wrenches elsewhere survive.
        const int dragged =
          pert.active && pert.select > 0 && pert.select < m->nbody ? pert.select : -1;
        if (impl->pert_body >= 0 && impl->pert_body != dragged)
            mju_zero(d->xfrc_applied + 6 * impl->pert_body, 6);
        impl->pert_body = dragged;
        if (dragged >= 0) {
            mjv_applyPerturbPose(m, d, &pert, 0);
            mjv_applyPerturbForce(m, d, &pert);
        }
        inject_noise(impl, m, d);
        studio::ForEachPlugin<studio::ModelPlugin>([&](studio::ModelPlugin *plugin) {
            if (plugin->pre_step) plugin->pre_step(plugin, m, d);
        });
        impl->stepped = true;
    } else {
        // Paused, as in Studio, a drag moves the selected tree's free root (or mocap body) to the
        // drag target; step() then recomputes the frames.
        if (pert.active && pert.select > 0 && pert.select < m->nbody)
            mjv_applyPerturbPose(m, d, &pert, 1);
        *integrate           = false;
        impl->speed_wall0    = {};
        impl->speed_measured = -1.0;
    }
    if (impl->recording) record(impl, env, cam, pert);
    publish(impl, env);
    return true;
}

static void close_window(ViewerImpl *impl)
{
    impl->stop = true;
    if (impl->thread.joinable()) impl->thread.join();
}

// The window's side

/* Picture-in-picture views and screenshots, drawn by a renderer of their own. Studio's App draws
 * them with a second Render() of the window's renderer, and each mjrf_render frees the material
 * instances it did not use (MaterialManager::RemoveUnusedMaterials): the UI's, which the UI still
 * draws with, so Studio itself aborts on Picture-in-Picture in MuJoCo 4.0.0. A Filament context
 * of its own keeps these materials apart, and each view is a Render() of its own, since a render
 * reads back one target. */
struct OffscreenRenderer
{
    std::unique_ptr<mujoco::Renderer>   renderer;
    std::unique_ptr<studio::ModelScene> scene;
    std::uint64_t                       model_gen = 0;
};

// A picture-in-picture view in a window of its own; `id` keeps the window's identity while its
// camera, and so its title, changes.
struct CameraView
{
    int              id = 0;
    studio::PipState pip;
};

// The window's Filament objects.
struct Gfx
{
    studio::Window      *window   = nullptr;
    mujoco::Renderer    *renderer = nullptr;
    mujoco::ImguiBridge *bridge   = nullptr;
};

struct SystemFont
{
    std::string label; // "Family Style"
    std::string file;
    int         index = 0; // face in a collection
};

/* The Style Editor's edits, kept over the Studio theme and saved in the settings file. Studio's
 * App re-applies its theme every frame while the editor is closed, so edits never outlive it;
 * here a theme is applied when it is picked or the DPI scale changes, and the edits, the values
 * that differ from the theme, are applied on top of it. */
struct StyleState
{
    // Field name (Colors: "Col.<ImGui colour name>") -> its values; sizes per unit DPI scale.
    std::map<std::string, std::vector<float>> edits;
    std::string                               font_file; // the chosen UI font; empty: Studio's
    int                                       font_index = 0;
    std::string                               font_loaded; // the font io.FontDefault is
    std::map<std::string, ImFont *>           fonts;       // system fonts loaded, by key
    ImGuiStyle                                base; // the theme as applied: what edits differ from
    studio::GuiTheme        theme = studio::GuiTheme::kDark; // the theme base holds
    float                   dpi   = 0.0f;                    // the FontScaleDpi base was made at
    bool                    dirty = true; // base and edits to apply before a frame
    std::vector<SystemFont> system_fonts; // fontconfig's, listed on first use
    bool                    fonts_listed = false;
    ImGuiTextFilter         font_filter;
};

// What the window's thread keeps between frames: Studio's App state, less the simulation.
struct WindowState
{
    std::shared_ptr<mjModel>            model;
    std::uint64_t                       model_gen = 0;
    DataPtr                             data;
    std::unique_ptr<studio::ModelScene> scene;
    std::vector<mjvGeom>                overlays;
    std::vector<mjvGeom>                scene_geoms; // the extra geoms render() last drew
    mjvScene                            plugin_scene{};
    mjOption                            env_opt{};

    mjvCamera cam{};
    int       camera_idx = studio::kTumbleCameraIdx;
    float     cam_speed  = studio::kMinFlySpeed;

    studio::ViewGizmoAnimation gizmo_animation;
    mjvOption                  opt{};
    mjvPerturb                 pert{};
    mjtByte                    render_flags[mjNRNDFLAG]{};

    // Studio's panels drive these; the Env's thread does the stepping and keeps the history.
    studio::StepControl               step_control;
    studio::SimulationTimelineState   timeline;
    int                               env_history_index = 0;
    double                            speed_measured    = -1.0;
    int                               speed_index       = 0;
    int                               key_idx           = -1;
    int                               nthread           = 0;
    bool                              update_threadpool = false;
    studio::SimProfiler               profiler;
    std::array<mjTimerStat, mjNTIMER> timer_prev{};

    int              frame_pending         = 0; // frames left to fit the start camera to the layout
    int              framing_frames        = 0; // Viewer::framing_frames
    float            min_panel_width       = 0.0f;     // Viewer::min_panel_width [font heights]
    double           joint_limit_margin    = 0.05;     // Viewer::joint_limit_margin
    bool             joint_limit_highlight = true;     // Viewer::joint_limit_highlight
    bool             manipulability_ellipsoid = false; // Viewer::manipulability_ellipsoid
    float            manipulability_scale     = 0.3f;  // Viewer::manipulability_scale [m]
    std::string      ini_path;
    studio::GuiTheme default_theme = studio::GuiTheme::kDark; // Viewer::theme
    studio::GuiTheme theme         = studio::GuiTheme::kDark;
    StyleState       style;
    float            font_scale = 1.0f;
    bool             quit       = false;

    // App::UiTempState's windows.
    bool help               = false;
    bool info               = false;
    bool profiler_open      = false;
    bool profiler_show_iter = false;
    bool options_panel      = true;
    enum DragMode { kDragNone, kDragForce, kDragTorque };
    int  drag_mode       = kDragNone; // the Perturb section's
    bool inspector_panel = true;
    bool toolbar         = false;
    bool view_gizmo      = true;
    bool status_bar      = false;
    bool full_screen     = false;
    bool console         = false;
    bool style_editor    = false;
    bool filament_editor = false;

    std::vector<CameraView>                    camera_views;
    int                                        next_camera_view = 1; // its window's id
    std::map<ImTextureID, std::pair<int, int>> pip_sizes;            // each PiP texture's size
    std::vector<ImTextureID>                   retired_textures;     // freed after render()
    OffscreenRenderer                          offscreen;            // PiP views, screenshots
    char                                       watch_field[1000] = "qpos";
    int                                        watch_index       = 0;
    std::vector<mjtNum>                        state;
    int                                        state_sig = 0;

    std::shared_ptr<mjSpec> spec;
    mjsElement             *element        = nullptr; // selected in the Explorer
    bool                    focus_explorer = false;   // a pick wants the Explorer tab in front
    bool                    reveal_element = false;   // and its row opened and scrolled to
    // Bodies and sites whose frame is drawn (Frames), by name, so the choice survives a rebuild.
    std::set<std::pair<mjtObj, std::string>> shown_frames;
    ImGuiTextFilter                          frame_filter;
    // The Data tab's Frame (RViz's Fixed Frame): a body or site by name; mjOBJ_UNKNOWN is world.
    std::pair<mjtObj, std::string> fixed_frame{ mjOBJ_UNKNOWN, "" };
    ImGuiTextFilter                fixed_frame_filter;
    float                          explorer_split = -1.0f;
    int                            element_mode   = 0; // 0 spec, 1 model, 2 data
    // The element tab's right-click Copy: the row the menu opened on, and a copy asked for.
    int  copy_row     = -1;
    bool copy_pending = false;

    enum FileDialog {
        kNoDialog,
        kSaveXml,
        kSaveMjb,
        kSaveScreenshot,
        kPrintModel,
        kPrintData,
        kLoadConfig
    };
    FileDialog file_dialog    = kNoDialog;
    char       filename[1024] = "";

    char                     record_path[1024] = "";
    int                      record_camera     = kRecordCurrent;
    int                      record_resolution = 0; // index into kRecordResolutions
    int                      record_fps        = 0;
    bool                     recording         = false;
    std::string              record_status;
    std::vector<RobotStatus> robots;
    double                   fps    = 0.0;
    float                    gpu_ms = 0.0f;

    bool paused() const { return step_control.GetPauseState() != PauseState::kUnpaused; }
};

/* MuJoCo's default free camera, aimed at what the scene shows now rather than at mjModel's
 * statistics, which are taken at qpos0 and count the infinite floor: the bounding sphere of
 * every geom but the planes, at the model's azimuth and elevation. The scene is drawn across
 * the whole window, but the panels cover its sides, so the sphere must fit the free part:
 * `visible_w` x `visible_h` of a window `window_h` high, which the vertical fov spans. */
static void frame_scene(
  const mjModel *m,
  const mjData  *d,
  mjvCamera     *cam,
  float          visible_w,
  float          visible_h,
  float          window_h
)
{
    mjv_defaultFreeCamera(m, cam);
    mjtNum lo[3] = { mjMAXVAL, mjMAXVAL, mjMAXVAL };
    mjtNum hi[3] = { -mjMAXVAL, -mjMAXVAL, -mjMAXVAL };
    bool   any   = false;
    for (int g = 0; g < m->ngeom; ++g) {
        if (m->geom_type[g] == mjGEOM_PLANE) continue;
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], d->geom_xpos[3 * g + k] - m->geom_rbound[g]);
            hi[k] = std::max(hi[k], d->geom_xpos[3 * g + k] + m->geom_rbound[g]);
        }
        any = true;
    }
    if (!any) return;
    mjtNum half[3];
    for (int k = 0; k < 3; ++k) {
        cam->lookat[k] = 0.5 * (lo[k] + hi[k]);
        half[k]        = 0.5 * (hi[k] - lo[k]);
    }
    const mjtNum radius   = mju_norm3(half);
    const mjtNum tan_half = mju_tan(0.5 * m->vis.global.fovy * mjPI / 180.0);
    const mjtNum tan_fit  = tan_half * std::min(visible_w, visible_h) / std::max(window_h, 1.0f);
    cam->distance         = radius / mju_sin(std::atan(tan_fit));
}

static void take_snapshot(ViewerImpl *impl, WindowState *ws, mujoco::Renderer *renderer)
{
    std::shared_ptr<mjModel> model;
    bool                     got_data = false;
    {
        const std::lock_guard<std::mutex> lock(impl->mtx);
        Snapshot                         &snap = impl->snapshot;
        if (snap.model_gen != ws->model_gen) {
            model         = snap.model;
            ws->spec      = snap.spec;
            ws->element   = nullptr; // it pointed into the old spec
            ws->model_gen = snap.model_gen;
            // The old data belongs to the old model: it never goes back to the Env.
            ws->data = snap.fresh ? std::move(snap.data) : DataPtr(mj_makeData(model.get()));
            got_data = snap.fresh;
            if (snap.fresh) snap.data = DataPtr(mj_makeData(model.get()));
            snap.fresh = false;
        } else if (snap.fresh) {
            std::swap(ws->data, snap.data);
            snap.fresh = false;
            got_data   = true;
        }
        if (!impl->want_overlays) {
            ws->overlays        = std::move(snap.overlays);
            impl->want_overlays = true;
        }
        ws->env_opt                = snap.opt;
        ws->robots                 = snap.robots;
        ws->recording              = snap.recording;
        ws->record_status          = snap.record_status;
        ws->speed_measured         = snap.speed_measured;
        ws->env_history_index      = snap.timeline.history_index;
        ws->timeline.sim_head_time = snap.timeline.sim_head_time;
        ws->timeline.SetHistorySize(snap.timeline.history_size);
        if (!ws->timeline.scrubber_active) ws->timeline.SetHistoryIndex(ws->env_history_index);
        if (impl->camera_request) {
            ws->cam = *impl->camera_request;
            ws->camera_idx =
              ws->cam.type == mjCAMERA_FIXED ? ws->cam.fixedcamid : studio::kTumbleCameraIdx;
            impl->camera_request.reset();
        }
    }
    if (model) {
        const bool first = !ws->model;
        ws->scene.reset();
        ws->model = std::move(model);
        if (!got_data) mj_forward(ws->model.get(), ws->data.get());
        ws->scene = std::make_unique<studio::ModelScene>(
          renderer->GetContext(), renderer->GetScene("scene"), ws->model.get(), studio::DrawTextAt
        );
        mjv_defaultPerturb(&ws->pert);
        ws->profiler.Clear();
        ws->timer_prev = {};
        // Framed in build_gui, once the docking layout says how much of the window is free.
        if (first && ws->cam.type == mjCAMERA_FREE && ws->cam.distance <= 0.0)
            ws->frame_pending = ws->framing_frames;
    }
    if (got_data) ws->profiler.Update(ws->model.get(), ws->data.get());
}

// An ImGuiStyle field the Style Editor edits; `scaled` ones are sizes ScaleAllSizes() scales.
struct StyleField
{
    enum Kind { kFloat, kVec2, kInt, kBool };
    const char *name;
    std::size_t offset;
    Kind        kind;
    bool        scaled;
};

static_assert(sizeof(ImGuiDir) == sizeof(int));
#define MJK_STYLE_FIELD(field, kind, scaled) \
    StyleField { #field, offsetof(ImGuiStyle, field), StyleField::kind, scaled }

/* Every ImGuiStyle field of ImGui 1.92.9 but FontScaleDpi, which follows the monitor, the colours
 * and the internals; `scaled` as in ImGuiStyle::ScaleAllSizes(). */
static constexpr StyleField kStyleFields[] = {
    MJK_STYLE_FIELD(FontSizeBase, kFloat, false),
    MJK_STYLE_FIELD(FontScaleMain, kFloat, false),
    MJK_STYLE_FIELD(Alpha, kFloat, false),
    MJK_STYLE_FIELD(DisabledAlpha, kFloat, false),
    MJK_STYLE_FIELD(WindowPadding, kVec2, true),
    MJK_STYLE_FIELD(WindowRounding, kFloat, true),
    MJK_STYLE_FIELD(WindowBorderSize, kFloat, true),
    MJK_STYLE_FIELD(WindowBorderHoverPadding, kFloat, true),
    MJK_STYLE_FIELD(WindowMinSize, kVec2, true),
    MJK_STYLE_FIELD(WindowTitleAlign, kVec2, false),
    MJK_STYLE_FIELD(WindowMenuButtonPosition, kInt, false),
    MJK_STYLE_FIELD(ChildRounding, kFloat, true),
    MJK_STYLE_FIELD(ChildBorderSize, kFloat, true),
    MJK_STYLE_FIELD(PopupRounding, kFloat, true),
    MJK_STYLE_FIELD(PopupBorderSize, kFloat, true),
    MJK_STYLE_FIELD(FramePadding, kVec2, true),
    MJK_STYLE_FIELD(FrameRounding, kFloat, true),
    MJK_STYLE_FIELD(FrameBorderSize, kFloat, true),
    MJK_STYLE_FIELD(ItemSpacing, kVec2, true),
    MJK_STYLE_FIELD(ItemInnerSpacing, kVec2, true),
    MJK_STYLE_FIELD(CellPadding, kVec2, true),
    MJK_STYLE_FIELD(TouchExtraPadding, kVec2, true),
    MJK_STYLE_FIELD(IndentSpacing, kFloat, true),
    MJK_STYLE_FIELD(ColumnsMinSpacing, kFloat, true),
    MJK_STYLE_FIELD(ScrollbarSize, kFloat, true),
    MJK_STYLE_FIELD(ScrollbarRounding, kFloat, true),
    MJK_STYLE_FIELD(ScrollbarPadding, kFloat, true),
    MJK_STYLE_FIELD(GrabMinSize, kFloat, true),
    MJK_STYLE_FIELD(GrabRounding, kFloat, true),
    MJK_STYLE_FIELD(LogSliderDeadzone, kFloat, true),
    MJK_STYLE_FIELD(ImageRounding, kFloat, true),
    MJK_STYLE_FIELD(ImageBorderSize, kFloat, true),
    MJK_STYLE_FIELD(TabRounding, kFloat, true),
    MJK_STYLE_FIELD(TabBorderSize, kFloat, true),
    MJK_STYLE_FIELD(TabMinWidthBase, kFloat, true),
    MJK_STYLE_FIELD(TabMinWidthShrink, kFloat, true),
    MJK_STYLE_FIELD(TabCloseButtonMinWidthSelected, kFloat, false),
    MJK_STYLE_FIELD(TabCloseButtonMinWidthUnselected, kFloat, false),
    MJK_STYLE_FIELD(TabBarBorderSize, kFloat, true),
    MJK_STYLE_FIELD(TabBarOverlineSize, kFloat, true),
    MJK_STYLE_FIELD(TableAngledHeadersAngle, kFloat, false),
    MJK_STYLE_FIELD(TableAngledHeadersTextAlign, kVec2, false),
    MJK_STYLE_FIELD(TreeLinesFlags, kInt, false),
    MJK_STYLE_FIELD(TreeLinesSize, kFloat, true),
    MJK_STYLE_FIELD(TreeLinesRounding, kFloat, true),
    MJK_STYLE_FIELD(MenuItemRounding, kFloat, true),
    MJK_STYLE_FIELD(SelectableRounding, kFloat, true),
    MJK_STYLE_FIELD(DragDropTargetRounding, kFloat, true),
    MJK_STYLE_FIELD(DragDropTargetBorderSize, kFloat, true),
    MJK_STYLE_FIELD(DragDropTargetPadding, kFloat, true),
    MJK_STYLE_FIELD(ColorMarkerSize, kFloat, true),
    MJK_STYLE_FIELD(ColorButtonPosition, kInt, false),
    MJK_STYLE_FIELD(ButtonTextAlign, kVec2, false),
    MJK_STYLE_FIELD(SelectableTextAlign, kVec2, false),
    MJK_STYLE_FIELD(InputTextCursorSize, kFloat, true),
    MJK_STYLE_FIELD(SeparatorSize, kFloat, true),
    MJK_STYLE_FIELD(SeparatorTextBorderSize, kFloat, true),
    MJK_STYLE_FIELD(SeparatorTextAlign, kVec2, false),
    MJK_STYLE_FIELD(SeparatorTextPadding, kVec2, true),
    MJK_STYLE_FIELD(DisplayWindowPadding, kVec2, true),
    MJK_STYLE_FIELD(DisplaySafeAreaPadding, kVec2, true),
    MJK_STYLE_FIELD(DockingNodeHasCloseButton, kBool, false),
    MJK_STYLE_FIELD(DockingSeparatorSize, kFloat, true),
    MJK_STYLE_FIELD(MouseCursorScale, kFloat, true),
    MJK_STYLE_FIELD(AntiAliasedLines, kBool, false),
    MJK_STYLE_FIELD(AntiAliasedLinesUseTex, kBool, false),
    MJK_STYLE_FIELD(AntiAliasedFill, kBool, false),
    MJK_STYLE_FIELD(CurveTessellationTol, kFloat, false),
    MJK_STYLE_FIELD(CircleTessellationMaxError, kFloat, false),
    MJK_STYLE_FIELD(HoverStationaryDelay, kFloat, false),
    MJK_STYLE_FIELD(HoverDelayShort, kFloat, false),
    MJK_STYLE_FIELD(HoverDelayNormal, kFloat, false),
    MJK_STYLE_FIELD(HoverFlagsForTooltipMouse, kInt, false),
    MJK_STYLE_FIELD(HoverFlagsForTooltipNav, kInt, false),
};
#undef MJK_STYLE_FIELD

static constexpr const char *kStyleColorPrefix = "Col.";

// A field's values, a size divided by the DPI scale so it reads the same on every monitor.
static std::vector<float> style_values(const ImGuiStyle &s, const StyleField &f, float dpi)
{
    const char *p     = reinterpret_cast<const char *>(&s) + f.offset;
    const float scale = f.scaled && dpi > 0.0f ? dpi : 1.0f;
    switch (f.kind) {
    case StyleField::kFloat:
        return { *reinterpret_cast<const float *>(p) / scale };
    case StyleField::kVec2: {
        const ImVec2 v = *reinterpret_cast<const ImVec2 *>(p);
        return { v.x / scale, v.y / scale };
    }
    case StyleField::kInt:
        return { static_cast<float>(*reinterpret_cast<const int *>(p)) };
    case StyleField::kBool:
        return { *reinterpret_cast<const bool *>(p) ? 1.0f : 0.0f };
    }
    return {};
}

static void
  set_style_values(ImGuiStyle *s, const StyleField &f, const std::vector<float> &v, float dpi)
{
    char             *p     = reinterpret_cast<char *>(s) + f.offset;
    const float       scale = f.scaled && dpi > 0.0f ? dpi : 1.0f;
    const std::size_t n     = f.kind == StyleField::kVec2 ? 2 : 1;
    if (v.size() != n) return;
    switch (f.kind) {
    case StyleField::kFloat:
        *reinterpret_cast<float *>(p) = v[0] * scale;
        break;
    case StyleField::kVec2:
        *reinterpret_cast<ImVec2 *>(p) = ImVec2(v[0] * scale, v[1] * scale);
        break;
    case StyleField::kInt:
        *reinterpret_cast<int *>(p) = static_cast<int>(v[0]);
        break;
    case StyleField::kBool:
        *reinterpret_cast<bool *>(p) = v[0] != 0.0f;
        break;
    }
}

// Sets one saved edit, a field or a colour by name; unknown names, from another ImGui, are skipped.
static void
  apply_style_edit(ImGuiStyle *s, const std::string &key, const std::vector<float> &v, float dpi)
{
    if (key.rfind(kStyleColorPrefix, 0) == 0) {
        const std::string_view name = std::string_view(key).substr(std::strlen(kStyleColorPrefix));
        for (int c = 0; c < ImGuiCol_COUNT; ++c)
            if (name == ImGui::GetStyleColorName(c) && v.size() == 4)
                s->Colors[c] = ImVec4(v[0], v[1], v[2], v[3]);
        return;
    }
    for (const StyleField &f : kStyleFields)
        if (key == f.name) set_style_values(s, f, v, dpi);
}

/* The theme from scratch, as Studio's window starts ImGui (a default style scaled to the
 * monitor, then SetupTheme), with the edits on top. Starting from scratch keeps an earlier edit
 * of a field SetupTheme leaves alone out of the base the edits are taken against. */
static void apply_style(StyleState *st, studio::GuiTheme theme)
{
    ImGuiStyle &s   = ImGui::GetStyle();
    const float dpi = s.FontScaleDpi;
    s               = ImGuiStyle();
    s.ScaleAllSizes(dpi);
    s.FontScaleDpi = dpi;
    // ImGui sets a zero FontSizeBase from the default font on the next frame; set it now, so
    // the base holds it too.
    ImGuiIO      &io   = ImGui::GetIO();
    const ImFont *font = io.FontDefault            ? io.FontDefault
                         : io.Fonts->Fonts.empty() ? nullptr
                                                   : io.Fonts->Fonts[0];
    if (font && font->LegacySize > 0.0f) s.FontSizeBase = font->LegacySize;
    studio::SetupTheme(theme);
    st->base  = s;
    st->theme = theme;
    st->dpi   = dpi;
    for (const auto &[key, values] : st->edits) apply_style_edit(&s, key, values, dpi);
    // A FontSizeBase set mid-frame is undone by the next PopFont(); ImGui's editor goes this way.
    s._NextFrameFontSizeBase = s.FontSizeBase;
    st->dirty                = false;
}

// The values the style now differs from the theme in; a change is saved with the settings.
static void capture_style_edits(StyleState *st)
{
    const ImGuiStyle                         &s = ImGui::GetStyle();
    std::map<std::string, std::vector<float>> edits;
    for (const StyleField &f : kStyleFields) {
        std::vector<float> now = style_values(s, f, st->dpi);
        if (now != style_values(st->base, f, st->dpi)) edits[f.name] = std::move(now);
    }
    for (int c = 0; c < ImGuiCol_COUNT; ++c) {
        const ImVec4 &now = s.Colors[c], &was = st->base.Colors[c];
        if (now.x != was.x || now.y != was.y || now.z != was.z || now.w != was.w)
            edits[std::string(kStyleColorPrefix) + ImGui::GetStyleColorName(c)] = {
                now.x, now.y, now.z, now.w
            };
    }
    if (edits != st->edits) {
        st->edits = std::move(edits);
        ImGui::MarkIniSettingsDirty();
    }
}

// Studio's icon font merged into a UI font, as AddStudioFonts (studio/ux/fonts.cc) adds it.
static constexpr float   kStudioFontSize     = 16.0f;
static constexpr float   kStudioIconSize     = 13.0f;
static constexpr ImWchar kStudioIconRanges[] = { 0xf000, 0xf3ff, 0 };

/* Makes the chosen font io.FontDefault, loading it the first time with Studio's icons merged in;
 * between frames, so the atlas never changes under a frame. An empty choice is Studio's font. */
static void load_ui_font(StyleState *st)
{
    const std::string want =
      st->font_file.empty() ? "" : std::to_string(st->font_index) + ":" + st->font_file;
    if (want == st->font_loaded) return;
    ImGuiIO &io   = ImGui::GetIO();
    ImFont  *font = nullptr;
    if (!want.empty()) {
        if (const auto it = st->fonts.find(want); it != st->fonts.end()) {
            font = it->second;
        } else if (std::error_code ec; std::filesystem::is_regular_file(st->font_file, ec)) {
            ImFontConfig config;
            config.FontNo          = st->font_index;
            const std::string stem = std::filesystem::path(st->font_file).stem().string();
            ImStrncpy(config.Name, stem.c_str(), IM_COUNTOF(config.Name));
            font = io.Fonts->AddFontFromFileTTF(st->font_file.c_str(), kStudioFontSize, &config);
            const std::vector<std::byte> icons = studio::LoadFontResource(studio::kIconFontFile);
            if (font && !icons.empty()) {
                void *copy = IM_ALLOC(icons.size()); // the atlas frees it
                std::memcpy(copy, icons.data(), icons.size());
                ImFontConfig merge;
                merge.MergeMode = true;
                ImStrncpy(merge.Name, studio::kIconFontFile, IM_COUNTOF(merge.Name));
                io.Fonts->AddFontFromMemoryTTF(
                  copy, static_cast<int>(icons.size()), kStudioIconSize, &merge, kStudioIconRanges
                );
            }
            if (font) st->fonts[want] = font;
        }
        if (!font) {
            MJ_LOG_ERROR("the font '" << st->font_file << "' could not be loaded");
            st->font_file.clear();
        }
    }
    io.FontDefault  = font; // null: the atlas's first font, Studio's
    st->font_loaded = font ? want : "";
    st->dirty       = true; // FontSizeBase follows the default font
}

/* Studio adds its fonts from memory without names, so ImGui's font list shows them blank; they
 * are named after their files, in the order AddStudioFonts adds them: main, icons merged, mono. */
static void name_studio_fonts()
{
    ImFontAtlas *atlas      = ImGui::GetIO().Fonts;
    const char  *unmerged[] = { studio::kMainFontFile, studio::kMonoFontFile };
    std::size_t  next       = 0;
    for (ImFontConfig &source : atlas->Sources) {
        if (source.Name[0] != '\0') continue;
        const char *file = source.MergeMode             ? studio::kIconFontFile
                           : next < std::size(unmerged) ? unmerged[next++]
                                                        : nullptr;
        if (!file) continue;
        const std::string stem = std::filesystem::path(file).stem().string();
        ImStrncpy(source.Name, stem.c_str(), IM_COUNTOF(source.Name));
    }
}

// The scalable fonts fontconfig knows, by family and style; variable-font instances are left
// out, as ImGui's rasteriser draws a font's default instance only.
static std::vector<SystemFont> list_system_fonts()
{
    std::vector<SystemFont> fonts;
    FcConfig               *config = FcInitLoadConfigAndFonts();
    if (!config) return fonts;
    FcPattern   *pattern = FcPatternCreate();
    FcObjectSet *objects =
      FcObjectSetBuild(FC_FAMILY, FC_STYLE, FC_FILE, FC_INDEX, FC_FONTFORMAT, nullptr);
    if (FcFontSet *set = FcFontList(config, pattern, objects)) {
        for (int i = 0; i < set->nfont; ++i) {
            FcChar8 *family = nullptr, *style = nullptr, *file = nullptr, *format = nullptr;
            int      index = 0;
            if (FcPatternGetString(set->fonts[i], FC_FILE, 0, &file) != FcResultMatch
                || FcPatternGetString(set->fonts[i], FC_FAMILY, 0, &family) != FcResultMatch)
                continue;
            FcPatternGetString(set->fonts[i], FC_STYLE, 0, &style);
            FcPatternGetInteger(set->fonts[i], FC_INDEX, 0, &index);
            FcPatternGetString(set->fonts[i], FC_FONTFORMAT, 0, &format);
            const std::string_view fmt = format ? reinterpret_cast<const char *>(format) : "";
            if ((fmt != "TrueType" && fmt != "CFF") || (index >> 16) != 0) continue;
            std::string label = reinterpret_cast<const char *>(family);
            if (style) label += std::string(" ") + reinterpret_cast<const char *>(style);
            fonts.push_back({ std::move(label), reinterpret_cast<const char *>(file), index });
        }
        FcFontSetDestroy(set);
    }
    FcObjectSetDestroy(objects);
    FcPatternDestroy(pattern);
    FcConfigDestroy(config);
    std::ranges::sort(fonts, {}, &SystemFont::label);
    const auto dupes = std::ranges::unique(fonts, {}, &SystemFont::label);
    fonts.erase(dupes.begin(), dupes.end());
    return fonts;
}

/* The style's own section of ImGui's settings, which Studio's SaveSettings writes with the rest
 * of ImGui's: "[MjKdl][Style]", a line per edit ("FrameRounding=3", "Col.Text=1,1,1,1") and
 * "Font=<face index>,<file>". */
static void register_style_settings(StyleState *st)
{
    ImGuiSettingsHandler handler;
    handler.TypeName   = "MjKdl";
    handler.TypeHash   = ImHashStr("MjKdl");
    handler.UserData   = st;
    handler.ReadInitFn = [](ImGuiContext *, ImGuiSettingsHandler *h) {
        auto *s = static_cast<StyleState *>(h->UserData);
        s->edits.clear();
        s->font_file.clear();
        s->font_index = 0;
    };
    handler.ReadOpenFn = [](ImGuiContext *, ImGuiSettingsHandler *h, const char *name) -> void * {
        return std::string_view(name) == "Style" ? h->UserData : nullptr;
    };
    handler.ReadLineFn = [](ImGuiContext *, ImGuiSettingsHandler *, void *entry, const char *text) {
        auto                  *s    = static_cast<StyleState *>(entry);
        const std::string_view line = text;
        const std::size_t      eq   = line.find('=');
        if (eq == std::string_view::npos) return;
        const std::string_view key   = line.substr(0, eq);
        std::string_view       value = line.substr(eq + 1);
        if (key == "Font") {
            const std::size_t comma = value.find(',');
            if (comma == std::string_view::npos) return;
            s->font_index = std::atoi(std::string(value.substr(0, comma)).c_str());
            s->font_file  = std::string(value.substr(comma + 1));
            return;
        }
        std::vector<float> values;
        while (!value.empty()) {
            const std::size_t comma = value.find(',');
            values.push_back(std::strtof(std::string(value.substr(0, comma)).c_str(), nullptr));
            value.remove_prefix(comma == std::string_view::npos ? value.size() : comma + 1);
        }
        s->edits[std::string(key)] = std::move(values);
    };
    handler.ApplyAllFn = [](ImGuiContext *, ImGuiSettingsHandler *h) {
        static_cast<StyleState *>(h->UserData)->dirty = true;
    };
    handler.WriteAllFn = [](ImGuiContext *, ImGuiSettingsHandler *h, ImGuiTextBuffer *out) {
        const auto *s = static_cast<const StyleState *>(h->UserData);
        if (s->edits.empty() && s->font_file.empty()) return;
        out->appendf("[%s][Style]\n", h->TypeName);
        for (const auto &[key, values] : s->edits) {
            out->appendf("%s=", key.c_str());
            for (std::size_t i = 0; i < values.size(); ++i)
                out->appendf(i == 0 ? "%.9g" : ",%.9g", values[i]);
            out->append("\n");
        }
        if (!s->font_file.empty())
            out->appendf("Font=%d,%s\n", s->font_index, s->font_file.c_str());
        out->append("\n");
    };
    ImGui::AddSettingsHandler(&handler);
}

/* Studio's LoadSettings, with the style's edits read afresh: an empty or reset file has no
 * section, and ImGui reads nothing from it. */
static void load_settings(WindowState *ws)
{
    ws->style.edits.clear();
    ws->style.font_file.clear();
    ws->style.font_index = 0;
    ws->theme            = studio::LoadSettings(ws->ini_path, ws->default_theme);
    ws->style.theme      = ws->theme;
    ws->style.dirty      = true;
}

/* Before a frame: the chosen font, then the theme with the edits when they are due. A theme
 * picked since (View -> Theme, the Style Editor's Theme) starts afresh, without the edits. */
static void prepare_style(WindowState *ws)
{
    StyleState *st = &ws->style;
    if (ws->theme != st->theme) {
        st->edits.clear();
        st->dirty = true;
        ImGui::MarkIniSettingsDirty();
    }
    load_ui_font(st);
    if (ImGui::GetStyle().FontScaleDpi != st->dpi) st->dirty = true;
    if (st->dirty) apply_style(st, ws->theme);
}

// The Style Editor: Studio's themes and the system fonts, then ImGui's editor.
static void style_editor_gui(WindowState *ws)
{
    StyleState *st = &ws->style;
    ImGui::SeparatorText("Studio");
    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);
    if (ImGui::BeginCombo("Theme", studio::ThemeName(ws->theme))) {
        for (const studio::GuiTheme theme :
             { studio::GuiTheme::kLight, studio::GuiTheme::kDark, studio::GuiTheme::kClassic })
            if (ImGui::Selectable(studio::ThemeName(theme), ws->theme == theme)) ws->theme = theme;
        ImGui::EndCombo();
    }
    ImGui::SetItemTooltip("%s", "Starts afresh from this theme; edits below are dropped");

    const auto        current = std::ranges::find_if(st->system_fonts, [st](const SystemFont &f) {
        return f.file == st->font_file && f.index == st->font_index;
    });
    const std::string preview = st->font_file.empty()               ? "Studio"
                                : current != st->system_fonts.end() ? current->label
                                                                    : st->font_file;
    if (ImGui::BeginCombo("Font", preview.c_str(), ImGuiComboFlags_HeightLarge)) {
        if (!st->fonts_listed) {
            st->system_fonts = list_system_fonts();
            st->fonts_listed = true;
        }
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        st->font_filter.Draw("##font_filter", -FLT_MIN);
        if (ImGui::Selectable("Studio", st->font_file.empty())) {
            st->font_file.clear();
            st->font_index = 0;
            ImGui::MarkIniSettingsDirty();
        }
        for (const SystemFont &font : st->system_fonts) {
            if (!st->font_filter.PassFilter(font.label.c_str())) continue;
            const bool chosen = font.file == st->font_file && font.index == st->font_index;
            if (ImGui::Selectable(font.label.c_str(), chosen) && !chosen) {
                st->font_file  = font.file;
                st->font_index = font.index;
                ImGui::MarkIniSettingsDirty();
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SetItemTooltip("%s", "System fonts (fontconfig), with Studio's icons merged in");
    ImGui::PopItemWidth();
    ImGui::TextDisabled("%s", "Edits are kept over the theme and saved with the settings.");

    ImGui::ShowStyleEditor();
    capture_style_edits(st);
}

/* Selects the model element type/id in the Explorer, with the Explorer tab in front and its
 * branch opened and scrolled to; the selection is cleared when the spec has no such element. */
static void select_element(WindowState *ws, mjtObj type, int id)
{
    ws->element = nullptr;
    if (id < 0 || !ws->spec) return;
    mjSpec *spec = ws->spec.get();
    for (mjsElement *e = mjs_firstElement(spec, type); e; e = mjs_nextElement(spec, e)) {
        if (mjs_getId(e) != id) continue;
        ws->element         = e;
        ws->focus_explorer  = true;
        ws->reveal_element  = true;
        ws->inspector_panel = true;
        return;
    }
}

static void handle_mouse(WindowState *ws, const studio::Window &window)
{
    ImGuiIO &io = ImGui::GetIO();
    // The scene's context menu captures the mouse while open; a click on the scene still acts,
    // so the second click of a right double-click centres the view as the menu closes.
    const bool over_scene = ImGui::IsPopupOpen("SceneContextMenu")
                            && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);
    if (io.WantCaptureMouse && !over_scene) return;
    const mjModel *m = ws->model.get();
    mjData        *d = ws->data.get();

    const float mouse_x  = io.MousePos.x / io.DisplaySize.x;
    const float mouse_y  = io.MousePos.y / io.DisplaySize.y;
    const float mouse_dx = io.MouseDelta.x / io.DisplaySize.x;
    const float mouse_dy = io.MouseDelta.y / io.DisplaySize.y;
    const bool  any_down = ImGui::IsMouseDown(ImGuiMouseButton_Left)
                          || ImGui::IsMouseDown(ImGuiMouseButton_Right)
                          || ImGui::IsMouseDown(ImGuiMouseButton_Middle);
    const bool dragging = (mouse_dx != 0.0f || mouse_dy != 0.0f) && any_down;
    if (!any_down) ws->pert.active = 0;

    // Ctrl + drag pushes the selected body, as in Studio; so does a plain left drag in the
    // Perturb section's Force or Torque mode, which leaves the camera where it is.
    const bool mode_drag = !io.KeyCtrl && ws->drag_mode != WindowState::kDragNone
                           && ImGui::IsMouseDown(ImGuiMouseButton_Left) && ws->pert.select > 0;
    if (dragging && mode_drag) {
        const bool       force  = ws->drag_mode == WindowState::kDragForce;
        const mjtMouse   action = force ? (io.KeyShift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V)
                                        : (io.KeyShift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V);
        const mjtPertBit active = force ? mjPERT_TRANSLATE : mjPERT_ROTATE;
        if (active != ws->pert.active) studio::InitPerturb(m, d, &ws->cam, &ws->pert, active);
        studio::MovePerturb(m, d, &ws->cam, &ws->pert, action, mouse_dx, mouse_dy);
    } else if (dragging && io.KeyCtrl && ws->pert.select > 0) {
        mjtMouse action = mjMOUSE_NONE;
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            if (io.KeyAlt)
                action = io.KeyShift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
            else
                action = io.KeyShift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
        } else if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            action = io.KeyShift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
        } else if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
            action = mjMOUSE_ZOOM;
        }
        const mjtPertBit active =
          (action == mjMOUSE_MOVE_V || action == mjMOUSE_MOVE_H) ? mjPERT_TRANSLATE : mjPERT_ROTATE;
        if (active != ws->pert.active) studio::InitPerturb(m, d, &ws->cam, &ws->pert, active);
        studio::MovePerturb(m, d, &ws->cam, &ws->pert, action, mouse_dx, mouse_dy);
    }

    if (!mode_drag)
        studio::MoveCameraWithMouse(
          m, &ws->cam, ws->camera_idx == studio::kFreeCameraIdx, studio::ReadImGuiCameraInput()
        );

    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        const studio::PickResult picked =
          studio::Pick(m, d, &ws->cam, mouse_x, mouse_y, window.GetAspectRatio(), &ws->opt);
        // As in Studio, the picked body is also the Explorer's element, its properties shown.
        select_element(ws, mjOBJ_BODY, picked.body);
        if (picked.body >= 0) {
            ws->pert.select     = picked.body;
            ws->pert.flexselect = picked.flex;
            ws->pert.skinselect = picked.skin;
            mjtNum offset[3];
            mju_sub3(offset, picked.point, d->xpos + 3 * picked.body);
            mju_mulMatTVec(ws->pert.localpos, d->xmat + 9 * picked.body, offset, 3, 3);
        } else {
            mjv_defaultPerturb(&ws->pert);
        }
    }
    // A right click that did not drag (a drag pans) opens the context menu, which has something
    // to offer only while a body or element is selected.
    const float drag = io.MouseDragThreshold;
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)
        && io.MouseDragMaxDistanceSqr[ImGuiMouseButton_Right] < drag * drag
        && (ws->element || ws->pert.select > 0))
        ImGui::OpenPopup("SceneContextMenu");
    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Right)) {
        ImGui::ClosePopupsExceptModals();
        const studio::PickResult picked =
          studio::Pick(m, d, &ws->cam, mouse_x, mouse_y, window.GetAspectRatio(), &ws->opt);
        mju_copy3(ws->cam.lookat, picked.point);
        if (picked.body > 0 && io.KeyCtrl) {
            ws->cam.type        = mjCAMERA_TRACKING;
            ws->cam.trackbodyid = picked.body;
            ws->cam.fixedcamid  = -1;
            ws->camera_idx      = studio::kTrackingCameraIdx;
        }
    }
}

static void toggle_pause(WindowState *ws)
{
    ws->step_control.SetPauseState(
      ws->paused() ? PauseState::kUnpaused : PauseState::kNormalPaused
    );
}

static void request_reset(WindowState *ws, Commands *cmd)
{
    cmd->reset    = true;
    cmd->keyframe = ws->key_idx;
}

static void align_camera(WindowState *ws)
{
    const mjModel *m      = ws->model.get();
    const int      cam_id = m->vis.global.cameraid;
    if (cam_id >= 0 && cam_id < m->ncam)
        ws->camera_idx = studio::SetCamera(m, &ws->cam, cam_id);
    else
        mjv_defaultFreeCamera(m, &ws->cam);
}

// Studio's App::LoadHistory: the scrubber moves, and the simulation pauses on that frame.
static void load_history(WindowState *ws, int index)
{
    ws->timeline.SetHistoryIndex(index);
    ws->step_control.SetPauseState(PauseState::kNormalPaused);
}

static void toggle_profiler(WindowState *ws)
{
    if (!ws->profiler_open) {
        ws->profiler_open      = true;
        ws->profiler_show_iter = false;
    } else if (!ws->profiler_show_iter) {
        ws->profiler_show_iter = true;
    } else {
        ws->profiler_open = false;
    }
}

static void toggle_window(bool &open)
{
    open                               = !open;
    ImGui::GetIO().WantSaveIniSettings = true;
}

static void toggle_flag(mjtByte &flag) { flag = flag ? 0 : 1; }

static void set_font_scale(WindowState *ws, float scale)
{
    const float old = ws->font_scale;
    ws->font_scale  = std::clamp(scale, 0.5f, 3.0f);
    studio::RescaleDock(ws->font_scale / old);
}

/* Studio's App::HandleKeyboardEvents. Every key reaches key_pressed(); a key the caller captured
 * does nothing in the viewer. Open Model, Unload and the editor's Delete have no Env to act on. */
static void handle_keys(ViewerImpl *impl, WindowState *ws, Commands *cmd)
{
    const bool typing = ImGui::GetIO().WantCaptureKeyboard;
    for (int k = ImGuiKey_NamedKey_BEGIN; k < ImGuiKey_NamedKey_END; ++k) {
        const bool down = !typing && ImGui::IsKeyDown(static_cast<ImGuiKey>(k));
        impl->keys[key_index(k)].store(down, std::memory_order_relaxed);
    }
    if (typing) return;
    const auto chord = [impl](ImGuiKeyChord c) {
        const int k = key_index(c & ~ImGuiMod_Mask_);
        return (k < 0 || !impl->captured[k].load(std::memory_order_relaxed))
               && studio::ImGui_IsChordJustPressed(c);
    };
    constexpr auto kCtrlShift = ImGuiMod_Ctrl | ImGuiMod_Shift;
    const mjModel *m          = ws->model.get();
    mjvOption     &vis        = ws->opt;
    const bool     freecam    = ws->camera_idx == studio::kFreeCameraIdx;

    if (chord(ImGuiKey_S | kCtrlShift)) {
        ws->file_dialog = WindowState::kSaveMjb;
    } else if (chord(ImGuiKey_S | ImGuiMod_Ctrl)) {
        ws->file_dialog = WindowState::kSaveXml;
    } else if (chord(ImGuiKey_M | ImGuiMod_Ctrl)) {
        ws->file_dialog = WindowState::kPrintModel;
    } else if (chord(ImGuiKey_D | ImGuiMod_Ctrl)) {
        ws->file_dialog = WindowState::kPrintData;
    } else if (chord(ImGuiKey_P | ImGuiMod_Ctrl)) {
        ws->file_dialog = WindowState::kSaveScreenshot;
    } else if (chord(ImGuiKey_C | ImGuiMod_Ctrl)) {
        studio::MaybeSaveToClipboard(studio::KeyframeToString(m, ws->data.get(), false));
    } else if (chord(ImGuiKey_L | ImGuiMod_Ctrl)) {
        cmd->reload = true;
    } else if (chord(ImGuiKey_Q | ImGuiMod_Ctrl)) {
        ws->quit = true;
    } else if (chord(ImGuiKey_A | ImGuiMod_Ctrl)) {
        align_camera(ws);
    } else if (chord(ImGuiKey_Tab | ImGuiMod_Shift)) {
        ws->inspector_panel = !ws->inspector_panel;
    } else if (chord(ImGuiKey_Tab)) {
        ws->options_panel = !ws->options_panel;
    } else if (chord(ImGuiKey_Minus | ImGuiMod_Ctrl)) {
        set_font_scale(ws, ws->font_scale - 0.1f);
    } else if (chord(ImGuiKey_Equal | ImGuiMod_Ctrl)) {
        set_font_scale(ws, ws->font_scale + 0.1f);
    } else if (chord(ImGuiKey_Minus)) {
        studio::SetSpeedIndex(&ws->step_control, ws->speed_index, ws->speed_index + 1);
    } else if (chord(ImGuiKey_Equal)) {
        studio::SetSpeedIndex(&ws->step_control, ws->speed_index, ws->speed_index - 1);
    } else if (chord(ImGuiKey_LeftArrow)) {
        if (ws->paused()) load_history(ws, ws->timeline.history_index - 1);
    } else if (chord(ImGuiKey_RightArrow)) {
        if (ws->paused()) {
            if (ws->timeline.history_index == 0)
                ws->step_control.RequestSingleStep();
            else
                load_history(ws, ws->timeline.history_index + 1);
        }
    } else if (chord(ImGuiKey_Space)) {
        toggle_pause(ws);
    } else if (chord(ImGuiKey_Backspace)) {
        request_reset(ws, cmd);
    } else if (chord(ImGuiKey_PageUp)) {
        if (ws->pert.select > 0) {
            ws->pert.select     = m->body_parentid[ws->pert.select];
            ws->pert.flexselect = -1;
            ws->pert.skinselect = -1;
            if (ws->pert.select <= 0) ws->pert.active = 0;
        }
    } else if (chord(ImGuiKey_F1)) {
        toggle_window(ws->help);
    } else if (chord(ImGuiKey_F2)) {
        toggle_window(ws->info);
    } else if (chord(ImGuiKey_F3)) {
        toggle_profiler(ws);
    } else if (chord(ImGuiKey_F6)) {
        vis.frame = (vis.frame + 1) % mjNFRAME;
    } else if (chord(ImGuiKey_F7)) {
        vis.label = (vis.label + 1) % mjNLABEL;
    } else if (chord(ImGuiKey_F11)) {
        ws->full_screen = !ws->full_screen;
    } else if (chord(ImGuiKey_H)) {
        toggle_flag(vis.flags[mjVIS_CONVEXHULL]);
    } else if (chord(ImGuiKey_X)) {
        toggle_flag(vis.flags[mjVIS_TEXTURE]);
    } else if (chord(ImGuiKey_J)) {
        toggle_flag(vis.flags[mjVIS_JOINT]);
    } else if (!freecam && chord(ImGuiKey_Q)) {
        toggle_flag(vis.flags[mjVIS_CAMERA]);
    } else if (chord(ImGuiKey_U)) {
        toggle_flag(vis.flags[mjVIS_ACTUATOR]);
    } else if (chord(ImGuiKey_Comma)) {
        toggle_flag(vis.flags[mjVIS_ACTIVATION]);
    } else if (chord(ImGuiKey_Z)) {
        toggle_flag(vis.flags[mjVIS_LIGHT]);
    } else if (chord(ImGuiKey_V)) {
        toggle_flag(vis.flags[mjVIS_TENDON]);
    } else if (chord(ImGuiKey_Y)) {
        toggle_flag(vis.flags[mjVIS_RANGEFINDER]);
    } else if (!freecam && chord(ImGuiKey_E)) {
        toggle_flag(vis.flags[mjVIS_CONSTRAINT]);
    } else if (chord(ImGuiKey_I)) {
        toggle_flag(vis.flags[mjVIS_INERTIA]);
    } else if (chord(ImGuiKey_Apostrophe)) {
        toggle_flag(vis.flags[mjVIS_SCLINERTIA]);
    } else if (chord(ImGuiKey_B)) {
        toggle_flag(vis.flags[mjVIS_PERTFORCE]);
    } else if (chord(ImGuiKey_O)) {
        toggle_flag(vis.flags[mjVIS_PERTOBJ]);
    } else if (chord(ImGuiKey_C)) {
        toggle_flag(vis.flags[mjVIS_CONTACTPOINT]);
    } else if (chord(ImGuiKey_N)) {
        toggle_flag(vis.flags[mjVIS_ISLAND]);
    } else if (chord(ImGuiKey_F)) {
        toggle_flag(vis.flags[mjVIS_CONTACTFORCE]);
    } else if (chord(ImGuiKey_P)) {
        toggle_flag(vis.flags[mjVIS_CONTACTSPLIT]);
    } else if (chord(ImGuiKey_T)) {
        toggle_flag(vis.flags[mjVIS_TRANSPARENT]);
    } else if (!freecam && chord(ImGuiKey_A)) {
        toggle_flag(vis.flags[mjVIS_AUTOCONNECT]);
    } else if (chord(ImGuiKey_M)) {
        toggle_flag(vis.flags[mjVIS_COM]);
    } else if (!freecam && chord(ImGuiKey_D)) {
        toggle_flag(vis.flags[mjVIS_STATIC]);
    } else if (!freecam && chord(ImGuiKey_W)) {
        toggle_flag(ws->render_flags[mjRND_WIREFRAME]);
    } else if (chord(ImGuiKey_Semicolon)) {
        toggle_flag(vis.flags[mjVIS_SKIN]);
    } else if (chord(ImGuiKey_GraveAccent)) {
        toggle_flag(vis.flags[mjVIS_BODYBVH]);
    } else if (chord(ImGuiKey_Backslash)) {
        toggle_flag(vis.flags[mjVIS_MESHBVH]);
    } else if (chord(ImGuiKey_Escape)) {
        ws->camera_idx = studio::SetCamera(m, &ws->cam, studio::kTumbleCameraIdx);
    } else if (chord(ImGuiKey_LeftBracket)) {
        ws->camera_idx = studio::SetCamera(m, &ws->cam, ws->camera_idx - 1);
    } else if (chord(ImGuiKey_RightBracket)) {
        ws->camera_idx = studio::SetCamera(m, &ws->cam, ws->camera_idx + 1);
    } else {
        bool group_key = false;
        for (int g = 0; g < 6 && !group_key; ++g) {
            const ImGuiKey key = static_cast<ImGuiKey>(ImGuiKey_0 + g);
            if (chord(key | ImGuiMod_Shift)) {
                toggle_flag(vis.sitegroup[g]);
                group_key = true;
            } else if (chord(key)) {
                toggle_flag(vis.geomgroup[g]);
                group_key = true;
            }
        }
        // Unless WASD/QE flew the fly camera, offer the keys to plugins.
        if (!group_key
            && (!freecam || !studio::MoveCameraWithKeys(m, &ws->cam, studio::ReadImGuiCameraInput(), &ws->cam_speed))) {
            studio::ForEachPlugin<studio::KeyHandlerPlugin>([&](studio::KeyHandlerPlugin *p) {
                if (p->key_chord && p->on_key_pressed && chord(p->key_chord)) p->on_key_pressed(p);
            });
        }
    }
}

// The scene of `model_scene`, a scene named "scene" of `renderer`, through `camera`.
static mujoco::Renderer::View scene_view(
  const WindowState  *ws,
  mujoco::Renderer   *renderer,
  studio::ModelScene *model_scene,
  const mjvCamera    *camera
)
{
    const mjModel         *m = ws->model.get();
    const mjtByte         *f = ws->render_flags;
    mujoco::Renderer::View view;
    view.scene  = renderer->GetSceneHandle("scene");
    view.camera = mjv_camera2GLCamera(m, ws->data.get(), camera);
    if (f[mjRND_SEGMENT])
        view.draw_mode =
          f[mjRND_IDCOLOR] ? mjDRAW_MODE_SEGMENTATION_BY_ID : mjDRAW_MODE_SEGMENTATION_BY_COLOR;
    else if (f[mjRND_DEPTH])
        view.draw_mode = mjDRAW_MODE_DEPTH;
    else if (f[mjRND_WIREFRAME])
        view.draw_mode = mjDRAW_MODE_WIREFRAME;
    else if (ws->opt.flags[mjVIS_ISLAND])
        view.draw_mode = mjDRAW_MODE_ISLANDS;
    view.haze            = f[mjRND_HAZE];
    view.skybox          = f[mjRND_SKYBOX];
    view.shadows         = f[mjRND_SHADOW];
    view.reflections     = f[mjRND_REFLECTION];
    view.post_processing = f[mjRND_POST_PROCESSING];
    if (m->vis.headlight.active) {
        view.headlight = mujoco::Renderer::View::Headlight();
        std::copy_n(m->vis.headlight.diffuse, 3, view.headlight->color);
        view.headlight->intensity = model_scene->GetHeadlightIntensity();
    }
    return view;
}

/* The scene through `camera` at width x height, drawn by the offscreen renderer and read back
 * at once; `draw_mode` replaces the window's when given. Its Filament objects, like the window's,
 * belong to the window's thread. */
static std::span<const std::byte> render_offscreen(
  WindowState               *ws,
  mjvCamera                 *camera,
  std::optional<mjrDrawMode> draw_mode,
  int                        width,
  int                        height
)
{
    OffscreenRenderer &off = ws->offscreen;
    mjModel           *m   = ws->model.get();
    if (!off.renderer) {
        mjrfContextConfig config;
        mjrf_defaultContextConfig(&config);
        config.graphics_api = mjGRAPHICS_API_OPENGL;
        off.renderer        = std::make_unique<mujoco::Renderer>(config);
        mjrfRenderTargetConfig target;
        mjrf_defaultRenderTargetConfig(&target);
        target.color_format = mjPIXEL_FORMAT_RGB8;
        target.depth_format = mjPIXEL_FORMAT_DEPTH32F;
        off.renderer->CreateRenderTarget("view", target);
    }
    if (!off.scene || off.model_gen != ws->model_gen) {
        off.scene.reset();
        off.scene = std::make_unique<studio::ModelScene>(
          off.renderer->GetContext(), off.renderer->GetScene("scene"), m
        );
        off.model_gen = ws->model_gen;
    }
    off.scene->Sync(
      m, ws->data.get(), { 0, 0, width, height }, &ws->opt, &ws->pert, camera, ws->scene_geoms
    );
    const auto target = off.renderer->GetRenderTargetHandle("view");
    off.renderer->ResizeRenderTarget(target, width, height);
    mujoco::Renderer::View view = scene_view(ws, off.renderer.get(), off.scene.get(), camera);
    if (draw_mode) view.draw_mode = *draw_mode;
    view.render_target = target;
    off.renderer->Render(std::span(&view, 1));
    return off.renderer->GetPixels(target);
}

// The next window opens as a tab beside the Explorer and Inspector, not floating.
static void dock_beside_inspector()
{
    if (const ImGuiWindow *inspector = ImGui::FindWindowByName("Inspector");
        inspector && inspector->DockId)
        ImGui::SetNextWindowDockID(inspector->DockId, ImGuiCond_Appearing);
}

// Studio's picture-in-picture modes: a fixed camera in colour (the window's wireframe or islands
// kept, as App draws it), depth or segmentation.
static mjrDrawMode pip_draw_mode(const WindowState *ws, studio::PipState::Mode mode)
{
    const mjtByte *f = ws->render_flags;
    if (mode == studio::PipState::Depth) return mjDRAW_MODE_DEPTH;
    if (mode == studio::PipState::Segmentation)
        return f[mjRND_IDCOLOR] ? mjDRAW_MODE_SEGMENTATION_BY_ID
                                : mjDRAW_MODE_SEGMENTATION_BY_COLOR;
    if (f[mjRND_WIREFRAME]) return mjDRAW_MODE_WIREFRAME;
    if (ws->opt.flags[mjVIS_ISLAND]) return mjDRAW_MODE_ISLANDS;
    return mjDRAW_MODE_DEFAULT;
}

// The view's image, drawn now at width x height into its texture.
static void
  upload_camera_view(WindowState *ws, const Gfx &gfx, studio::PipState *pip, int width, int height)
{
    if (pip->camera >= ws->model->ncam || width <= 0 || height <= 0) return;
    mjvCamera camera;
    mjv_defaultCamera(&camera);
    camera.type       = mjCAMERA_FIXED;
    camera.fixedcamid = pip->camera;
    const auto pixels = render_offscreen(ws, &camera, pip_draw_mode(ws, pip->mode), width, height);
    if (pixels.size() != static_cast<std::size_t>(width) * height * 3) return;
    /* ImguiBridge::UploadImage replaces a texture whose size changed under the same id and
     * destroys the old one at once, while the UI may still draw it. A new size gets a new id; the
     * old texture is freed after this frame's render(). */
    ImTextureID &texture = pip->texture;
    const auto   size    = ws->pip_sizes.find(texture);
    if (texture != ImTextureID_Invalid && size != ws->pip_sizes.end()
        && size->second != std::pair{ width, height }) {
        ws->retired_textures.push_back(texture);
        ws->pip_sizes.erase(size);
        texture = ImTextureID_Invalid;
    }
    texture = static_cast<ImTextureID>(gfx.bridge->UploadImage(texture, pixels, width, height));
    ws->pip_sizes[texture] = { width, height };
}

// Studio's PipGui row for one view (camera, mode), then its image at the window's aspect ratio.
static void camera_view_gui(WindowState *ws, const Gfx &gfx, studio::PipState *pip, float aspect)
{
    const mjModel *m           = ws->model.get();
    const float    avail       = ImGui::GetContentRegionAvail().x;
    const float    combo_width = (avail - ImGui::GetStyle().ItemSpacing.x) / 2;
    const auto     camera_name = [m](int i) {
        const char *name = mj_id2name(m, mjOBJ_CAMERA, i);
        return name ? std::string(name) : "Camera " + std::to_string(i);
    };
    ImGui::SetNextItemWidth(combo_width);
    if (ImGui::BeginCombo("##camera", camera_name(pip->camera).c_str())) {
        for (int i = 0; i < m->ncam; ++i)
            if (ImGui::Selectable(camera_name(i).c_str(), pip->camera == i)) pip->camera = i;
        ImGui::EndCombo();
    }
    static constexpr std::array<const char *, 3> kModes = { "Color", "Depth", "Segmentation" };
    ImGui::SameLine();
    ImGui::SetNextItemWidth(combo_width);
    if (ImGui::BeginCombo("##mode", kModes[pip->mode])) {
        for (int i = 0; i < static_cast<int>(kModes.size()); ++i)
            if (ImGui::Selectable(kModes[i], pip->mode == i))
                pip->mode = static_cast<studio::PipState::Mode>(i);
        ImGui::EndCombo();
    }
    const int width  = static_cast<int>(avail);
    const int height = aspect > 0.0f ? static_cast<int>(avail / aspect) : width;
    upload_camera_view(ws, gfx, pip, width, height);
    if (pip->texture != ImTextureID_Invalid)
        ImGui::Image(pip->texture, { static_cast<float>(width), static_cast<float>(height) });
}

// Studio's SaveToWebp (app.cc): a lossless WebP of an RGB888 image.
static void save_webp(int width, int height, const std::byte *rgb, const std::string &path)
{
    std::uint8_t *webp = nullptr;
    const size_t  size = WebPEncodeLosslessRGB(
      reinterpret_cast<const std::uint8_t *>(rgb), width, height, width * 3, &webp
    );
    if (size > 0) {
        std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(webp), size);
        MJ_LOG_INFO("screenshot saved to '" << path << "'");
    } else {
        MJ_LOG_ERROR("the screenshot could not be encoded");
    }
    WebPFree(webp);
}

static void simulation_gui(WindowState *ws, Commands *cmd)
{
    mjModel                     *m = ws->model.get();
    studio::SimulationGuiContext ctx;
    ctx.model             = m;
    ctx.data              = ws->data.get();
    ctx.step_control      = &ws->step_control;
    ctx.timeline          = &ws->timeline;
    ctx.speed_index       = &ws->speed_index;
    ctx.key_idx           = &ws->key_idx;
    ctx.nthread           = &ws->nthread;
    ctx.update_threadpool = &ws->update_threadpool;
    ctx.reset             = [ws, cmd] { request_reset(ws, cmd); };
    ctx.reload            = [cmd] { cmd->reload = true; };
    ctx.align             = [ws] { align_camera(ws); };
    ctx.save_keyframe     = [ws, cmd] { cmd->save_keyframe = ws->key_idx; };
    studio::SimulationGui(ctx);
    // The real-time factor the Env's thread measures; Studio's section has no slot for it.
    if (ws->speed_measured >= 0.0)
        ImGui::Text(
          "RTF %.2fx (target %.2fx)",
          ws->speed_measured / 100.0,
          ws->step_control.GetSpeed() / 100.0
        );
    else
        ImGui::TextDisabled("RTF -");
    ImGui::SetItemTooltip("%s", "Real-time factor: simulated time over wall time");
}

/* The Explorer's tree: Studio's SpecTreeGui (ux/gui_spec.cc), whose framed body rows never
 * show the selection, since ImGui draws a framed tree node alike whether selected or not. Here
 * the selected row takes the active header colour, and a body picked in the scene has its
 * branch opened and its row scrolled into view. */
struct SpecTree
{
    mjsElement           **element; // the selection, set by a click
    std::set<mjsElement *> open;    // bodies to open: the picked body's ancestors
    bool                   scroll = false;
};

// Studio's GetElementName: the name, or "(<type> <id>)" for an unnamed element.
static std::string element_label(mjsElement *element)
{
    std::string name = mjs_getString(mjs_getName(element));
    if (!name.empty()) return name;
    int id = mjs_getId(element);
    if (id == -1) {
        mjSpec *spec = mjs_getSpec(element);
        id           = 0;
        for (mjsElement *e = mjs_firstElement(spec, element->elemtype); e && e != element;
             e             = mjs_nextElement(spec, e))
            ++id;
    }
    return "(" + std::string(mju_type2Str(element->elemtype)) + " " + std::to_string(id) + ")";
}

// What an unnamed geom is, so "(geom 5)" says which: its mesh, or else its shape.
static std::string geom_note(mjsElement *element)
{
    static constexpr const char *kShapes[] = { "plane",   "hfield",    "sphere",
                                               "capsule", "ellipsoid", "cylinder",
                                               "box",     "mesh",      "sdf" };
    const mjsGeom               *geom      = mjs_asGeom(element);
    if (!geom) return {};
    const std::string mesh = mjs_getString(geom->meshname);
    if (geom->type == mjGEOM_MESH && !mesh.empty()) return "mesh " + mesh;
    return geom->type >= 0 && geom->type < (int)std::size(kShapes) ? kShapes[geom->type] : "";
}

/* A selected row in the theme's accent, the colour of an active button (the Inspector's S/M/D):
 * the header colours alone are too close to an unselected row to read at a glance. */
static constexpr int kSelectedColors = 3;
static void          push_selected_colors()
{
    const ImVec4 accent = ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive);
    ImGui::PushStyleColor(ImGuiCol_Header, accent);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, accent);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, accent);
}

// An element as an indented leaf of its group, highlighted when selected.
static void tree_element(SpecTree &tree, mjsElement *element)
{
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen
                               | ImGuiTreeNodeFlags_SpanAvailWidth
                               | ImGuiTreeNodeFlags_AllowOverlap;
    const bool selected = element == *tree.element;
    if (selected) {
        flags |= ImGuiTreeNodeFlags_Selected;
        push_selected_colors();
    }
    const std::string label = element_label(element);
    ImGui::TreeNodeEx(element, flags, "%s", label.c_str());
    if (selected) ImGui::PopStyleColor(kSelectedColors);
    if (ImGui::IsItemClicked()) *tree.element = element;
    if (element == *tree.element && tree.scroll) {
        ImGui::SetScrollHereY(0.5f);
        tree.scroll = false;
    }
    if (element->elemtype == mjOBJ_GEOM && mjs_getString(mjs_getName(element))[0] == '\0') {
        const std::string note = geom_note(element);
        if (!note.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("%s", note.c_str());
        }
    }
}

// A group of a body's elements: a dimmed header with its count, so it reads apart from bodies.
static void tree_body_children(SpecTree &tree, const char *heading, mjtObj type, mjsBody *body)
{
    mjsElement *child = mjs_firstChild(body, type, 0);
    if (!child) return;
    int count = 0;
    for (mjsElement *e = child; e; e = mjs_nextChild(body, e, 0)) ++count;
    constexpr ImGuiTreeNodeFlags flags =
      ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_DrawLinesFull;
    const std::string title =
      std::string(heading) + " (" + std::to_string(count) + ")###" + heading;
    mjsElement *selected = *tree.element;
    if (tree.scroll && selected && selected->elemtype == type && mjs_getParent(selected) == body)
        ImGui::SetNextItemOpen(true);
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    const bool open = studio::SectionHeader(title.c_str(), flags, 0.55f);
    ImGui::PopStyleColor();
    if (open) {
        for (; child; child = mjs_nextChild(body, child, 0)) tree_element(tree, child);
        ImGui::TreePop();
    }
}

static void tree_body(SpecTree &tree, mjsBody *body)
{
    const std::string label = element_label(body->element);
    ImGui::PushID(label.c_str());
    const bool selected = *tree.element == body->element;
    if (tree.open.count(body->element)) ImGui::SetNextItemOpen(true);
    if (selected) push_selected_colors();
    const bool open = studio::SectionHeader(
      label.c_str(),
      ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Framed
        | ImGuiTreeNodeFlags_DrawLinesFull | ImGuiTreeNodeFlags_AllowOverlap,
      0.55f
    );
    if (selected) ImGui::PopStyleColor(kSelectedColors);
    if (ImGui::IsItemClicked()) *tree.element = body->element;
    if (selected && tree.scroll) {
        ImGui::SetScrollHereY(0.5f);
        tree.scroll = false;
    }
    // The body's own elements come first, right under it, then its child bodies.
    if (open) {
        tree_body_children(tree, "Joints", mjOBJ_JOINT, body);
        tree_body_children(tree, "Geoms", mjOBJ_GEOM, body);
        tree_body_children(tree, "Sites", mjOBJ_SITE, body);
        tree_body_children(tree, "Frames", mjOBJ_FRAME, body);
        tree_body_children(tree, "Cameras", mjOBJ_CAMERA, body);
        tree_body_children(tree, "Lights", mjOBJ_LIGHT, body);
        for (mjsElement *child = mjs_firstChild(body, mjOBJ_BODY, 0); child;
             child             = mjs_nextChild(body, child, 0))
            tree_body(tree, mjs_asBody(child));
        ImGui::TreePop();
    }
    ImGui::PopID();
}

static void tree_element_list(SpecTree &tree, const char *heading, mjtObj type, mjSpec *spec)
{
    mjsElement *element = mjs_firstElement(spec, type);
    if (!element) return;
    constexpr ImGuiTreeNodeFlags flags =
      ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Framed;
    if (studio::SectionHeader(heading, flags, 0.55f)) {
        for (; element; element = mjs_nextElement(spec, element)) tree_element(tree, element);
        ImGui::TreePop();
    }
}

// Studio's SpecTreeGui, read-only, with `reveal` opening the selected element's branch.
static void spec_tree_gui(mjsElement **element, mjSpec *spec, bool reveal)
{
    SpecTree tree{ element, {}, reveal };
    if (reveal && *element) {
        for (mjsBody *b = mjs_getParent(*element); b; b = mjs_getParent(b->element)) {
            tree.open.insert(b->element);
            if (mjs_getId(b->element) == 0) break; // the world, its own parent
        }
    }
    const ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Framed;
    studio::ScopedStyle      style;
    style.Var(ImGuiStyleVar_ItemSpacing, ImVec2(4, 2));
    style.Var(ImGuiStyleVar_FramePadding, ImVec2(4, 0));
    style.Color(ImGuiCol_Border, ImGuiCol_WindowBg);

    if (tree.scroll) ImGui::SetNextItemOpen(true);
    if (studio::SectionHeader("Body Tree", flags, 0.65f)) {
        if (mjsElement *root = mjs_firstElement(spec, mjOBJ_BODY))
            tree_body(tree, mjs_asBody(root));
        ImGui::TreePop();
    }
    ImGui::PushID("$spec$");
    if (studio::SectionHeader("Elements", flags, 0.65f)) {
        tree_element_list(tree, "Actuators", mjOBJ_ACTUATOR, spec);
        tree_element_list(tree, "Sensors", mjOBJ_SENSOR, spec);
        tree_element_list(tree, "Flexes", mjOBJ_FLEX, spec);
        tree_element_list(tree, "Tendons", mjOBJ_TENDON, spec);
        tree_element_list(tree, "Pair", mjOBJ_PAIR, spec);
        tree_element_list(tree, "Exclude", mjOBJ_EXCLUDE, spec);
        tree_element_list(tree, "Equality", mjOBJ_EQUALITY, spec);
        tree_element_list(tree, "Numeric", mjOBJ_NUMERIC, spec);
        tree_element_list(tree, "Text", mjOBJ_TEXT, spec);
        tree_element_list(tree, "Tuple", mjOBJ_TUPLE, spec);
        tree_element_list(tree, "Key", mjOBJ_KEY, spec);
        tree_element_list(tree, "Default", mjOBJ_DEFAULT, spec);
        ImGui::TreePop();
    }
    if (studio::SectionHeader("Assets", flags, 0.65f)) {
        tree_element_list(tree, "Meshes", mjOBJ_MESH, spec);
        tree_element_list(tree, "Height Fields", mjOBJ_HFIELD, spec);
        tree_element_list(tree, "Skins", mjOBJ_SKIN, spec);
        tree_element_list(tree, "Textures", mjOBJ_TEXTURE, spec);
        tree_element_list(tree, "Materials", mjOBJ_MATERIAL, spec);
        ImGui::TreePop();
    }
    ImGui::PopID();
}

/* An element's world frame as mjData holds it, row-major rotation; a joint's is its anchor in
 * its body's orientation. False for elements without one. */
static bool element_world_frame(
  const mjModel *m,
  const mjData  *d,
  mjsElement    *element,
  const mjtNum **pos,
  const mjtNum **mat
)
{
    const int id = mjs_getId(element);
    switch (element->elemtype) {
    case mjOBJ_BODY:
        if (id < 0 || id >= m->nbody) return false;
        *pos = d->xpos + 3 * id, *mat = d->xmat + 9 * id;
        return true;
    case mjOBJ_JOINT:
        if (id < 0 || id >= m->njnt) return false;
        *pos = d->xanchor + 3 * id, *mat = d->xmat + 9 * m->jnt_bodyid[id];
        return true;
    case mjOBJ_GEOM:
        if (id < 0 || id >= m->ngeom) return false;
        *pos = d->geom_xpos + 3 * id, *mat = d->geom_xmat + 9 * id;
        return true;
    case mjOBJ_SITE:
        if (id < 0 || id >= m->nsite) return false;
        *pos = d->site_xpos + 3 * id, *mat = d->site_xmat + 9 * id;
        return true;
    case mjOBJ_CAMERA:
        if (id < 0 || id >= m->ncam) return false;
        *pos = d->cam_xpos + 3 * id, *mat = d->cam_xmat + 9 * id;
        return true;
    default:
        return false;
    }
}

// A body's or site's name, or Studio's "(body <id>)" when it has none.
static std::string frame_label(const mjModel *m, mjtObj type, int id)
{
    const char *name = mj_id2name(m, type, id);
    if (name && *name) return name;
    return std::string("(") + (type == mjOBJ_SITE ? "site " : "body ") + std::to_string(id) + ")";
}

// The Fixed Frame's id in this model, by name; -1 for world or a frame the model no longer has.
static int fixed_frame_id(const WindowState *ws)
{
    const auto &[type, name] = ws->fixed_frame;
    if (type != mjOBJ_BODY && type != mjOBJ_SITE) return -1;
    const mjModel *m     = ws->model.get();
    const int      count = type == mjOBJ_BODY ? m->nbody : m->nsite;
    for (int i = 0; i < count; ++i)
        if (frame_label(m, type, i) == name) return i;
    return -1;
}

// RViz's Fixed Frame: world or any body or site, under a name search.
static void fixed_frame_gui(WindowState *ws)
{
    const mjModel    *m       = ws->model.get();
    const bool        world   = fixed_frame_id(ws) < 0;
    const std::string preview = world ? "world" : ws->fixed_frame.second;
    ImGui::SetNextItemWidth(-ImGui::CalcTextSize("Frame").x - ImGui::GetStyle().ItemInnerSpacing.x);
    if (ImGui::BeginCombo("Frame", preview.c_str(), ImGuiComboFlags_HeightLarge)) {
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        ws->fixed_frame_filter.Draw("##fixed_frame_filter", -FLT_MIN);
        if (ImGui::Selectable("world", world)) ws->fixed_frame = { mjOBJ_UNKNOWN, "" };
        for (const auto &[type, count, heading] : { std::tuple{ mjOBJ_BODY, m->nbody, "Bodies" },
                                                    std::tuple{ mjOBJ_SITE, m->nsite, "Sites" } }) {
            if (count == 0) continue;
            ImGui::SeparatorText(heading);
            ImGui::PushID(heading);
            for (int i = type == mjOBJ_BODY ? 1 : 0; i < count; ++i) {
                std::string name = frame_label(m, type, i);
                if (!ws->fixed_frame_filter.PassFilter(name.c_str())) continue;
                const bool chosen = ws->fixed_frame.first == type && ws->fixed_frame.second == name;
                ImGui::PushID(i);
                if (ImGui::Selectable(name.c_str(), chosen)) ws->fixed_frame = { type, name };
                ImGui::PopID();
            }
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    ImGui::SetItemTooltip("%s", "The frame the rows below Studio's state the element's pose in");
}

/* The element's pose in the Fixed Frame (world when none): position [m], quaternion in the
 * wrapper's [x y z w] (docs/conventions.md) and roll, pitch, yaw [rad] about the frame's fixed
 * axes (ROS REP 103; KDL::Rotation::GetRPY). */
static void element_pose_gui(const WindowState *ws, mjsElement *element)
{
    const mjModel *m   = ws->model.get();
    const mjData  *d   = ws->data.get();
    const mjtNum  *pos = nullptr, *mat = nullptr;
    if (element->elemtype == mjOBJ_JOINT || !element_world_frame(m, d, element, &pos, &mat)) return;
    const int     frame       = fixed_frame_id(ws);
    const mjtNum  origin[3]   = { 0, 0, 0 };
    const mjtNum  identity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const bool    site        = ws->fixed_frame.first == mjOBJ_SITE;
    const mjtNum *fpos = frame < 0 ? origin : site ? d->site_xpos + 3 * frame : d->xpos + 3 * frame;
    const mjtNum *fmat = frame < 0 ? identity
                         : site    ? d->site_xmat + 9 * frame
                                   : d->xmat + 9 * frame;
    mjtNum        diff[3], rel_pos[3], wxyz[4];
    KDL::Rotation rel_rot; // row-major data[9], as MuJoCo's matrices
    mju_sub3(diff, pos, fpos);
    mju_mulMatTVec3(rel_pos, fmat, diff);
    mju_mulMatTMat(rel_rot.data, fmat, mat, 3, 3, 3);
    mju_mat2Quat(wxyz, rel_rot.data);
    const mjtNum xyzw[4] = { wxyz[1], wxyz[2], wxyz[3], wxyz[0] };
    double       rpy[3];
    rel_rot.GetRPY(rpy[0], rpy[1], rpy[2]);
    ImGui::SeparatorText("Pose");
    ImGui::PushID("rpy");
    {
        studio::ImGui_DataPtrTable table;
        table.DataPtr("pos", rel_pos, 0, 3);
        table.DataPtr("quat_xyzw", xyzw, 0, 4);
        table.DataPtr("rpy", rpy, 0, 3);
    }
    ImGui::PopID();
}

struct LoggedRow
{
    std::string label; // empty on a field's continuation rows (xmat's second and third)
    std::string value;
};

static std::string_view trimmed(std::string_view s)
{
    while (!s.empty() && s.front() == ' ') s.remove_prefix(1);
    while (!s.empty() && s.back() == ' ') s.remove_suffix(1);
    return s;
}

// The label and value rows ImGui logged for a two-column table: "|label|value|" per line.
static std::vector<LoggedRow> logged_rows(std::string_view log)
{
    std::vector<LoggedRow> rows;
    while (!log.empty()) {
        const std::size_t end  = log.find('\n');
        std::string_view  line = trimmed(log.substr(0, end));
        log.remove_prefix(end == std::string_view::npos ? log.size() : end + 1);
        if (line.empty() || line.front() != '|') continue;
        line.remove_prefix(1);
        if (!line.empty() && line.back() == '|') line.remove_suffix(1);
        const std::size_t split = line.find('|');
        if (split == std::string_view::npos) continue;
        rows.push_back({ std::string(trimmed(line.substr(0, split))),
                         std::string(trimmed(line.substr(split + 1))) });
    }
    return rows;
}

// The row of the table str_id (in the current ID scope) the mouse is over this frame, or -1.
static int hovered_table_row(const char *str_id)
{
    ImGuiTable *table = ImGui::TableFindByID(ImGui::GetID(str_id));
    return table ? ImGui::TableGetInstanceData(table, 0)->HoveredRowNext : -1;
}

static constexpr int kPoseRow = 1 << 20; // copy_row of the Pose table's rows, apart from Studio's

/* The S/M/D tab's tables, which Studio draws as plain text that cannot be selected: right-click
 * a field's row and Copy puts its values on the clipboard as shown, every line of a multi-line
 * field (xmat). The copy is captured with ImGui's logging the frame after the menu asks for it. */
static void element_tab_gui(WindowState *ws, mjsElement *element)
{
    static constexpr const char *kTable = "##PropertiesTable"; // Studio's ImGui_DataPtrTable
    const bool                   data   = ws->element_mode == 2;
    const bool                   copy   = ws->copy_pending;
    std::string                  studio_log, pose_log;
    // Data: the Frame and the pose in it first; Studio's every-mjData-field list folded below.
    int row = -1;
    // A joint's frame borrows its body's orientation; the Data tab shows its axis instead.
    const mjtNum *pose_pos = nullptr, *pose_mat = nullptr;
    const bool    pose =
      data && element->elemtype != mjOBJ_JOINT
      && element_world_frame(ws->model.get(), ws->data.get(), element, &pose_pos, &pose_mat);
    bool studio_now = true; // Studio's list is drawn this frame
    if (pose) {
        fixed_frame_gui(ws);
        if (copy) ImGui::LogToBuffer();
        element_pose_gui(ws, ws->element);
        if (copy) {
            pose_log = ImGui::GetCurrentContext()->LogBuffer.c_str();
            ImGui::LogFinish();
        }
        ImGui::PushID("rpy");
        if (const int pose_row = hovered_table_row(kTable); pose_row >= 0)
            row = kPoseRow + pose_row;
        ImGui::PopID();
        studio_now = ImGui::CollapsingHeader("All fields (mjData)");
    }
    if (studio_now) {
        if (copy) ImGui::LogToBuffer();
        if (ws->element_mode == 0)
            studio::ElementSpecGui(element);
        else if (ws->element_mode == 1)
            studio::ElementModelGui(ws->model.get(), ws->element);
        else
            studio::ElementDataGui(ws->data.get(), ws->element);
        if (copy) {
            studio_log = ImGui::GetCurrentContext()->LogBuffer.c_str();
            ImGui::LogFinish();
        }
        if (const int studio_row = hovered_table_row(kTable); studio_row >= 0) row = studio_row;
    }

    if (copy) {
        std::vector<LoggedRow> rows        = logged_rows(studio_log);
        const int              studio_rows = static_cast<int>(rows.size());
        for (LoggedRow &r : logged_rows(pose_log)) rows.push_back(std::move(r));
        int index = ws->copy_row >= kPoseRow ? studio_rows + ws->copy_row - kPoseRow : ws->copy_row;
        if (index >= 0 && index < static_cast<int>(rows.size())) {
            while (index > 0 && rows[index].label.empty()) --index; // the field's first row
            std::string text = rows[index].value;
            for (int i = index + 1; i < static_cast<int>(rows.size()) && rows[i].label.empty(); ++i)
                text += '\n' + rows[i].value;
            ImGui::SetClipboardText(text.c_str());
        }
        ws->copy_pending = false;
    }

    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)
        && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && row >= 0) {
        ws->copy_row = row;
        ImGui::OpenPopup("ElementCopy");
    }
    if (ImGui::BeginPopup("ElementCopy")) {
        if (ImGui::MenuItem("Copy")) ws->copy_pending = true;
        ImGui::EndPopup();
    }
}

static void explorer_gui(WindowState *ws)
{
    if (!ws->spec) {
        ImGui::Text("No mjSpec loaded.");
        return;
    }
    const ImVec2 region = ImGui::GetContentRegionAvail();
    if (region.y > 60.0f) {
        if (ws->explorer_split < 0) ws->explorer_split = region.y * 0.7f;
        ws->explorer_split = std::clamp(ws->explorer_split, 20.0f, region.y - 40.0f);
    }
    mjsElement *element = ws->element;
    bool        open    = element != nullptr;
    if (studio::ImGui_BeginHSplit("SpecExplorerTree", &ws->explorer_split, &open)) {
        spec_tree_gui(&element, ws->spec.get(), ws->reveal_element);
        ws->reveal_element = false;
        if (element != ws->element) {
            ws->element = element;
            if (element) {
                open                 = true;
                const int element_id = mjs_getId(element);
                if (element->elemtype == mjOBJ_BODY && ws->pert.select != element_id) {
                    mjv_defaultPerturb(&ws->pert);
                    ws->pert.select = element_id;
                }
            }
        }
        if (studio::ImGui_HSplit("SpecExplorerProperties", &ws->explorer_split, &open)) {
            ImGui::AlignTextToFramePadding();
            ImGui::Text("%s (%d)", mju_type2Str(ws->element->elemtype), mjs_getId(ws->element));
            // Square buttons a frame high, right-aligned by their real width, so a larger font
            // or DPI scale cannot crop them.
            const float button  = ImGui::GetFrameHeight();
            const float spacing = ImGui::GetStyle().ItemSpacing.x;
            ImGui::SameLine(ImGui::GetContentRegionMax().x - 3.0f * button - 2.0f * spacing);
            const char *labels[]   = { "S", "M", "D" };
            const char *tooltips[] = { "Spec", "Model", "Data" };
            for (int mode = 0; mode < 3; ++mode) {
                if (mode > 0) ImGui::SameLine();
                studio::ScopedStyle style;
                if (ws->element_mode == mode) style.Color(ImGuiCol_Button, ImGuiCol_ButtonActive);
                if (ImGui::Button(labels[mode], ImVec2(button, button))) ws->element_mode = mode;
                ImGui::SetItemTooltip("%s", tooltips[mode]);
            }
            // Studio shows only the type and id; the name, on its own line, says which one it is.
            if (const char *name = mjs_getString(mjs_getName(ws->element)); name && *name)
                ImGui::TextUnformatted(name);
            ImGui::Separator();
            element_tab_gui(ws, element);
        }
        if (!open) ws->element = nullptr;
    }
    studio::ImGui_EndHSplit(open);
}

// Studio's App::MainMenuGui. The Env owns the model, so Open, Unload, the spec Editor and the
// graphics-mode switch are shown but disabled.
static void main_menu_gui(WindowState *ws, Commands *cmd, const Gfx &gfx)
{
    const mjModel *m = ws->model.get();
    ImGui::PushStyleVar(
      ImGuiStyleVar_ItemSpacing,
      ImVec2(12.0f * ImGui::GetStyle().FontScaleDpi, ImGui::GetStyle().ItemSpacing.y)
    );
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            ImGui::MenuItem("Open Model File", "Ctrl+O", false, false);
            ImGui::Separator();
            if (ImGui::MenuItem("Save XML", "Ctrl+S")) ws->file_dialog = WindowState::kSaveXml;
            if (ImGui::MenuItem("Save MJB", "Ctrl+Shift+S"))
                ws->file_dialog = WindowState::kSaveMjb;
            if (ImGui::MenuItem("Save Screenshot", "Ctrl+P"))
                ws->file_dialog = WindowState::kSaveScreenshot;
            ImGui::Separator();
            if (ImGui::MenuItem("Print Model", "Ctrl+M"))
                ws->file_dialog = WindowState::kPrintModel;
            if (ImGui::MenuItem("Print Data", "Ctrl+D")) ws->file_dialog = WindowState::kPrintData;
            ImGui::Separator();
            ImGui::MenuItem("Unload", "Ctrl+U", false, false);
            ImGui::Separator();
            if (ImGui::MenuItem("Quit", "Ctrl+Q")) ws->quit = true;
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Simulation")) {
            if (ImGui::MenuItem("Pause", "Space", ws->paused())) toggle_pause(ws);
            if (ImGui::MenuItem("Reset", "Backspace")) request_reset(ws, cmd);
            if (ImGui::MenuItem("Reload", "Ctrl+L")) cmd->reload = true;
            if (m->nkey > 0) {
                ImGui::Separator();
                if (ImGui::BeginMenu("Keyframes")) {
                    if (ws->key_idx >= m->nkey || ws->key_idx < -1) ws->key_idx = -1;
                    const std::string key_name = studio::GetKeyframeName(m, ws->key_idx);
                    ImGui::SetNextItemWidth(200);
                    if (ImGui::BeginCombo("##Key", key_name.c_str())) {
                        if (ImGui::Selectable("\xE2\x80\x94", ws->key_idx == -1)) {
                            ws->key_idx = -1;
                            request_reset(ws, cmd);
                        }
                        for (int k = 0; k < m->nkey; k++) {
                            ImGui::PushID(k);
                            if (ImGui::Selectable(
                                  studio::GetKeyframeName(m, k).c_str(), ws->key_idx == k
                                )) {
                                ws->key_idx = k;
                                request_reset(ws, cmd);
                            }
                            ImGui::PopID();
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::BeginDisabled(ws->key_idx < 0 || ws->key_idx >= m->nkey);
                    if (ImGui::MenuItem("Save")) cmd->save_keyframe = ws->key_idx;
                    ImGui::EndDisabled();
                    if (ImGui::MenuItem("Copy"))
                        studio::MaybeSaveToClipboard(
                          studio::KeyframeToString(m, ws->data.get(), false)
                        );
                    ImGui::EndMenu();
                }
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            // Not in Studio's menu: the file in use, and loading another, saved there from then on.
            ImGui::MenuItem(ws->ini_path.c_str(), nullptr, false, false);
            if (ImGui::MenuItem("Load Config")) ws->file_dialog = WindowState::kLoadConfig;
            if (ImGui::MenuItem("Save Config")) studio::SaveSettings(ws->theme, ws->ini_path);
            if (ImGui::MenuItem("Reset Config")) {
                studio::ResetConfig(ws->ini_path);
                load_settings(ws);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Options", "Tab", ws->options_panel))
                ws->options_panel = !ws->options_panel;
            if (ImGui::MenuItem("Inspector", "Shift+Tab", ws->inspector_panel))
                ws->inspector_panel = !ws->inspector_panel;
            ImGui::MenuItem("Editor (\xEF\x83\x83)", nullptr, false, false);
            if (ImGui::MenuItem("Toolbar", nullptr, ws->toolbar)) ws->toolbar = !ws->toolbar;
            if (ImGui::MenuItem("View Gizmo", nullptr, ws->view_gizmo))
                ws->view_gizmo = !ws->view_gizmo;
            if (ImGui::MenuItem("Status Bar", nullptr, ws->status_bar))
                ws->status_bar = !ws->status_bar;
            if (ImGui::MenuItem("Full Screen", "F11", ws->full_screen))
                ws->full_screen = !ws->full_screen;
            ImGui::Separator();
            if (ImGui::MenuItem("Info", "F2", ws->info)) toggle_window(ws->info);
            if (ImGui::MenuItem("Profiler", "F3", ws->profiler_open)) toggle_profiler(ws);
            if (ImGui::MenuItem("Console", nullptr, ws->console)) toggle_window(ws->console);
            ImGui::Separator();
            if (ImGui::MenuItem("Add Camera View", nullptr, false, ws->model->ncam > 0))
                ws->camera_views.push_back({ .id = ws->next_camera_view++, .pip = {} });
            ImGui::Separator();
            studio::ThemeMenuGui(&ws->theme);
            if (ImGui::BeginMenu("Graphics Mode (Experimental)")) {
                const studio::GraphicsMode mode = gfx.window->GetGraphicsMode();
                ImGui::MenuItem(
                  "Filament OpenGL", nullptr, mode == studio::GraphicsMode::FilamentOpenGl, false
                );
                ImGui::MenuItem(
                  "Filament OpenGL Headless",
                  nullptr,
                  mode == studio::GraphicsMode::FilamentOpenGlHeadless,
                  false
                );
                ImGui::MenuItem(
                  "Filament OpenGL Software",
                  nullptr,
                  mode == studio::GraphicsMode::FilamentOpenGlSoftware,
                  false
                );
                ImGui::MenuItem(
                  "Filament Vulkan", nullptr, mode == studio::GraphicsMode::FilamentVulkan, false
                );
                ImGui::MenuItem(
                  "Filament Vulkan Software",
                  nullptr,
                  mode == studio::GraphicsMode::FilamentVulkanSoftware,
                  false
                );
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Plugins")) {
            // Placeholder, as in Studio: the plugins add their items below.
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("Help", "F1", ws->help)) toggle_window(ws->help);
            ImGui::Separator();
            if (ImGui::MenuItem("Style Editor", "", ws->style_editor))
                ws->style_editor = !ws->style_editor;
            ImGui::Separator();
            const std::string wrapper = std::string("mj_kdl_wrapper ") + version();
            const std::string mujoco  = std::string("MuJoCo ") + mj_versionString();
            ImGui::MenuItem(wrapper.c_str(), nullptr, false, false);
            ImGui::MenuItem(mujoco.c_str(), nullptr, false, false);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    ImGui::PopStyleVar();
}

// Studio's App::FileDialogGui for the File menu's items.
static void file_dialog_gui(WindowState *ws, Commands *cmd, const Gfx &gfx)
{
    if (ws->file_dialog == WindowState::kNoDialog) return;
    const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("FileDialog", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        const studio::DialogResult res = ws->file_dialog == WindowState::kLoadConfig
                                           ? studio::OpenFileDialog(ws->filename)
                                           : studio::SaveFileDialog(ws->filename);
        if (res.status == studio::DialogResult::kAccepted) {
            const mjModel *m = ws->model.get();
            switch (ws->file_dialog) {
            case WindowState::kSaveXml:
                cmd->save_xml = res.path;
                break;
            case WindowState::kSaveMjb:
                mj_saveModel(m, res.path.c_str(), nullptr, 0);
                break;
            case WindowState::kSaveScreenshot: {
                const int  width  = gfx.window->GetWidth();
                const int  height = gfx.window->GetHeight();
                const auto pixels = render_offscreen(ws, &ws->cam, std::nullopt, width, height);
                if (pixels.size() == static_cast<std::size_t>(width) * height * 3)
                    save_webp(width, height, pixels.data(), res.path);
                break;
            }
            case WindowState::kPrintModel:
                mj_printModel(m, res.path.c_str());
                break;
            case WindowState::kPrintData:
                mj_printData(m, ws->data.get(), res.path.c_str());
                break;
            case WindowState::kLoadConfig:
                ws->ini_path = res.path;
                load_settings(ws);
                break;
            default:
                break;
            }
        }
        if (res.status != studio::DialogResult::kError) {
            const std::size_t n = std::min(res.path.size(), sizeof(ws->filename) - 1);
            res.path.copy(ws->filename, n);
            ws->filename[n] = '\0';
        }
        if (res.status != studio::DialogResult::kPending) {
            ws->file_dialog = WindowState::kNoDialog;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    if (!ImGui::IsPopupOpen("FileDialog") && ws->file_dialog != WindowState::kNoDialog)
        ImGui::OpenPopup("FileDialog");
}

// Studio's App::ToolBarGui.
static void toolbar_gui(WindowState *ws, Commands *cmd)
{
    mjModel *m = ws->model.get();
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0, 0));
    if (ImGui::BeginTable("##ToolBarTable", 2)) {
        studio::ScopedStyle style;
        style.Var(
          ImGuiStyleVar_ItemSpacing,
          ImVec2(ImGui::GetStyle().ItemSpacing.x * 2.0f, ImGui::GetStyle().ItemSpacing.y)
        );
        const float label_width    = studio::GetExpectedLabelWidth();
        const float copy_btn_width = ImGui::GetFrameHeight();
        const float sp             = ImGui::GetStyle().ItemSpacing.x;
        const float right_width =
          copy_btn_width + label_width + sp + label_width + sp + label_width;
        const float separator_width = ImGui::GetFrameHeight() * .6f;

        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, right_width);
        ImGui::TableNextColumn();
        ImGui::Dummy(ImVec2(ImGui::GetStyle().WindowPadding.x, 0.0f));
        ImGui::SameLine(0.0f, 0.0f);

        const float  btn_size = ImGui::GetFrameHeight();
        const ImVec2 square_size(btn_size, btn_size);
        style.Var(ImGuiStyleVar_FrameRounding, 2.0f);
        if (ImGui::Button(studio::ICON_FA_REFRESH, square_size)) cmd->reload = true;
        ImGui::SetItemTooltip("%s", "Reload");
        ImGui::SameLine(0, 0.5 * separator_width);
        if (ImGui::Button(studio::ICON_FA_UNDO, square_size)) request_reset(ws, cmd);
        ImGui::SetItemTooltip("%s", "Reset");
        ImGui::SameLine(0, separator_width);
        studio::StepControlGui(&ws->step_control, ws->speed_index);
        ImGui::SameLine(0, separator_width);
        studio::TimelineScrubberGui(m, ws->data.get(), ws->step_control, ws->timeline);

        ImGui::TableNextColumn();
        studio::CameraSelectionGui(m, ws->data.get(), ws->cam, ws->camera_idx);
        ImGui::SameLine();
        studio::LabelSelectionGui(&ws->opt);
        ImGui::SameLine();
        studio::FrameSelectionGui(&ws->opt);
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();
}

// Studio's App::StatusBarGui.
static void status_bar_gui(WindowState *ws)
{
    if (ImGui::BeginTable("##StatusBarTable", 2)) {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableNextColumn();
        ImGui::Text(ws->paused() ? "Paused" : "Running");
        ImGui::TableNextColumn();
        studio::Console::Get().StatusGui(&ws->console);
        ImGui::EndTable();
    }
}

// Studio's App::HelpGui.
static void help_gui()
{
    const char *mod = ImGui::GetIO().ConfigMacOSXBehaviors ? "Cmd" : "Ctrl";

    const float pad    = ImGui::GetStyle().ItemSpacing.x;
    const float indent = pad * 2;
    const float col0   = ImGui::CalcTextSize("Toggle Fullscreen").x + pad;
    const float col1   = ImGui::CalcTextSize("Ctrl+R-Dblclick").x + pad + indent;
    const float col2   = ImGui::CalcTextSize("Center of Mass").x + pad + indent;
    const float col3   = ImGui::CalcTextSize("M").x + pad + indent;
    ImGui::Dummy(ImVec2(col0 + col1 + col2 + col3, 0));
    ImGui::Columns(4);
    ImGui::SetColumnWidth(0, col0);
    ImGui::SetColumnWidth(1, col1);
    ImGui::SetColumnWidth(2, col2);
    ImGui::SetColumnWidth(3, col3);

    const auto lines = [](std::initializer_list<const char *> texts) {
        for (const char *text : texts) ImGui::TextUnformatted(text);
    };
    ImGui::SeparatorText("Windows & UI");
    lines({ "Help",
            "Info",
            "Profiler",
            "Toggle Fullscreen",
            "Toggle Left UI",
            "Toggle Right UI",
            "Font Bigger",
            "Font Smaller" });
    ImGui::SeparatorText("Simulation");
    lines({ "Toggle Pause",
            "Reset Sim",
            "Step Back",
            "Step Forward",
            "Speed Up",
            "Speed Down",
            "Copy Keyframe" });
    ImGui::SeparatorText("Camera");
    lines({ "Tumble Camera",
            "Prev Camera",
            "Next Camera",
            "Align Camera",
            "Orbit Camera",
            "Pan Camera",
            "Zoom Camera",
            "Center Camera",
            "Track Camera" });
    ImGui::SeparatorText("Selection");
    lines({ "Select", "Select Parent", "Rotate Object", "Move Object" });
    ImGui::SeparatorText("Visualization");
    lines({ "Cycle Frames", "Cycle Labels", "Geom Group", "Site Group" });

    ImGui::NextColumn();
    ImGui::Indent(indent);
    ImGui::SeparatorText("");
    lines({ "F1", "F2", "F3", "F11", "Tab", "Sh+Tab" });
    ImGui::Text("%s+=", mod);
    ImGui::Text("%s+-", mod);
    ImGui::SeparatorText("");
    lines({ "Spc", "Bksp", "Left", "Right", "+", "-" });
    ImGui::Text("%s+C", mod);
    ImGui::SeparatorText("");
    lines({ "Esc", "[", "]" });
    ImGui::Text("%s+A", mod);
    lines({ "L-Drag", "[Sh+]R-Drag", "Scroll", "R-Dblclick" });
    ImGui::Text("%s+R-Dblclick", mod);
    ImGui::SeparatorText("");
    lines({ "Dblclick", "PgUp" });
    ImGui::Text("%s+L-Drag", mod);
    ImGui::Text("%s+R-Drag", mod);
    ImGui::SeparatorText("");
    lines({ "F6", "F7", "0-5", "Sh+0-5" });

    ImGui::NextColumn();
    ImGui::Indent(indent);
    ImGui::SeparatorText("Flags");
    lines({ "Activation",     "Actuator",      "Auto Connect",   "Body Tree",     "Camera",
            "Center of Mass", "Constraint",    "Contact Force",  "Contact Point", "Contact Split",
            "Convex Hull",    "Inertia",       "Island",         "Joint",         "Light",
            "Mesh Tree",      "Perturb Force", "Perturb Object", "Range Finder",  "Scale Inertia",
            "Skin",           "Static Body",   "Tendon",         "Texture",       "Transparent",
            "Wireframe" });

    ImGui::NextColumn();
    ImGui::Indent(indent);
    ImGui::SeparatorText("");
    lines({ ",", "U", "A",  "`", "Q", "M", "E", "F", "C", "P", "H", "I", "N",
            "J", "Z", "\\", "B", "O", "Y", "'", ";", "D", "V", "X", "T", "W" });
    ImGui::Columns();
}

// The Rendering section of Studio's App::ModelOptionsGui.
static void rendering_gui(WindowState *ws, float min_width)
{
    const mjModel *m = ws->model.get();
    ImGui::PushID("RenderingSection");
    char frame_label_tmp[32];
    std::snprintf(frame_label_tmp, sizeof(frame_label_tmp), " %s  Camera", studio::ICON_FA_CAMERA);
    const float combo_w =
      -ImGui::CalcTextSize(frame_label_tmp).x - ImGui::GetStyle().ItemInnerSpacing.x;

    {
        const std::string cam_name = studio::GetCameraName(m, ws->cam, ws->camera_idx);
        char              camera_text[32];
        std::snprintf(camera_text, sizeof(camera_text), "%s  Camera", studio::ICON_FA_CAMERA);
        ImGui::SetNextItemWidth(combo_w);
        if (ImGui::BeginCombo(camera_text, cam_name.c_str())) {
            const auto select = [&](int idx) {
                const std::string name = studio::GetCameraName(m, ws->cam, idx);
                if (ImGui::Selectable(name.c_str(), ws->camera_idx == idx))
                    ws->camera_idx = studio::SetCamera(m, &ws->cam, idx);
            };
            select(studio::kTumbleCameraIdx);
            select(studio::kFreeCameraIdx);
            select(studio::kTrackingCameraIdx);
            for (int c = 0; c < m->ncam; c++) select(c);
            ImGui::EndCombo();
        }
    }
    {
        char label_text[32];
        std::snprintf(label_text, sizeof(label_text), "%s  Label", studio::ICON_FA_COMMENT);
        ImGui::SetNextItemWidth(combo_w);
        if (ImGui::BeginCombo(label_text, mjLABELSTRING[ws->opt.label])) {
            for (int n = 0; n < mjNLABEL; n++)
                if (ImGui::Selectable(mjLABELSTRING[n], ws->opt.label == n)) ws->opt.label = n;
            ImGui::EndCombo();
        }
    }
    {
        char frame_text[32];
        std::snprintf(frame_text, sizeof(frame_text), "%s  Frame", studio::ICON_FA_ARROWS);
        ImGui::SetNextItemWidth(combo_w);
        if (ImGui::BeginCombo(frame_text, mjFRAMESTRING[ws->opt.frame])) {
            for (int n = 0; n < mjNFRAME; n++)
                if (ImGui::Selectable(mjFRAMESTRING[n], ws->opt.frame == n)) ws->opt.frame = n;
            ImGui::EndCombo();
        }
    }
    {
        char copy_cam_label[48];
        std::snprintf(
          copy_cam_label, sizeof(copy_cam_label), "%s  Copy Camera", studio::ICON_FA_COPY
        );
        if (ImGui::Button(copy_cam_label))
            studio::MaybeSaveToClipboard(studio::CameraToString(ws->data.get(), &ws->cam));
        ImGui::SetItemTooltip("%s", "Copy the current camera pose as an MJCF XML string");
    }
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::PopID();
    studio::RenderingGui(m, &ws->opt, ws->render_flags, min_width);
}

/* Every body and site with a checkbox that draws its frame, as RViz's TF display, under a name
 * search; sites carry the frames bodies do not, such as a robot's TCP and its F/T sensors. */
static void frames_gui(WindowState *ws)
{
    const mjModel *m = ws->model.get();
    ws->frame_filter.Draw("##frame_filter", -FLT_MIN);
    ImGui::SetItemTooltip("%s", "Filter by name (\"arm,-base\" includes and excludes)");
    // All and None act on the frames the filter shows.
    const bool all = ImGui::Button("All");
    ImGui::SameLine();
    const bool none = ImGui::Button("None");
    for (const auto &[type, count, heading] : { std::tuple{ mjOBJ_BODY, m->nbody, "Bodies" },
                                                std::tuple{ mjOBJ_SITE, m->nsite, "Sites" } }) {
        if (count == 0) continue;
        ImGui::SeparatorText(heading);
        ImGui::PushID(heading);
        for (int i = 0; i < count; ++i) {
            std::pair<mjtObj, std::string> key{ type, frame_label(m, type, i) };
            if (!ws->frame_filter.PassFilter(key.second.c_str())) continue;
            if (all) ws->shown_frames.insert(key);
            if (none) ws->shown_frames.erase(key);
            bool shown = ws->shown_frames.count(key) > 0;
            ImGui::PushID(i);
            if (ImGui::Checkbox("##shown", &shown)) {
                if (shown)
                    ws->shown_frames.insert(key);
                else
                    ws->shown_frames.erase(key);
            }
            // The name selects the element in the Explorer.
            ImGui::SameLine();
            const bool selected =
              ws->element && ws->element->elemtype == type && mjs_getId(ws->element) == i;
            if (selected) push_selected_colors();
            if (ImGui::Selectable(key.second.c_str(), selected)) select_element(ws, type, i);
            if (selected) ImGui::PopStyleColor(kSelectedColors);
            ImGui::PopID();
        }
        ImGui::PopID();
    }
}

// Within margin of either limit, as a part of the range; a joint without limits (+-inf) never is.
static bool joint_near_limit(const JointStatus &j, double margin)
{
    if (!(std::isfinite(j.lo) && std::isfinite(j.hi) && j.hi > j.lo)) return false;
    const double f = (j.q - j.lo) / (j.hi - j.lo);
    return f < margin || f > 1.0 - margin;
}

/* Yoshikawa's manipulability of the TCP's linear velocity (T. Yoshikawa, "Manipulability of
 * Robotic Mechanisms", IJRR 4(2), 1985): the TCP velocities unit-norm joint speeds reach form an
 * ellipsoid with the eigenvectors of Jv Jv^T as axes and the square roots of its eigenvalues as
 * radii; w = sqrt(det(Jv Jv^T)) is 0 at a singularity. Jv is MuJoCo's TCP Jacobian over the
 * robot's joints, the kinematics KDL's chain is built from. */
struct Manipulability
{
    mjtNum pos[3];   // TCP
    mjtNum axes[9];  // eigenvectors as columns, row-major
    mjtNum radii[3]; // [m/s per rad/s]
    mjtNum w = 0;
};

static bool
  robot_manipulability(const mjModel *m, const mjData *d, const RobotStatus &r, Manipulability *out)
{
    if (r.joints.empty()) return false;
    std::vector<mjtNum> jacp(3 * static_cast<std::size_t>(m->nv));
    const mjtNum       *tcp = nullptr;
    if (r.tcp_site >= 0 && r.tcp_site < m->nsite) {
        mj_jacSite(m, d, jacp.data(), nullptr, r.tcp_site);
        tcp = d->site_xpos + 3 * r.tcp_site;
    } else if (r.tip_body >= 0 && r.tip_body < m->nbody) {
        mj_jacBody(m, d, jacp.data(), nullptr, r.tip_body);
        tcp = d->xpos + 3 * r.tip_body;
    } else {
        return false;
    }
    mjtNum jjt[9] = {};
    for (const JointStatus &j : r.joints) {
        if (j.id < 0 || j.id >= m->njnt) return false;
        const int dof = m->jnt_dofadr[j.id];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                jjt[3 * a + b] += jacp[a * m->nv + dof] * jacp[b * m->nv + dof];
    }
    mjtNum eigval[3], quat[4];
    mju_eig3(eigval, out->axes, quat, jjt);
    for (int k = 0; k < 3; ++k) out->radii[k] = std::sqrt(std::max<mjtNum>(eigval[k], 0));
    out->w = out->radii[0] * out->radii[1] * out->radii[2];
    mju_copy3(out->pos, tcp);
    return true;
}

// A thin bar of the joint between its limits, its value centred on it.
static void joint_bar(const JointStatus &j, double margin, const char *unit)
{
    const bool near = joint_near_limit(j, margin);
    if (near) ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.85f, 0.2f, 0.2f, 1.0f));
    const float height = ImGui::GetTextLineHeight();
    ImGui::ProgressBar(
      static_cast<float>(std::clamp((j.q - j.lo) / (j.hi - j.lo), 0.0, 1.0)),
      ImVec2(-FLT_MIN, height),
      ""
    );
    if (near) ImGui::PopStyleColor();
    char value[32];
    std::snprintf(value, sizeof(value), "%.3f", j.q);
    const ImVec2 lo = ImGui::GetItemRectMin(), hi = ImGui::GetItemRectMax();
    const ImVec2 size = ImGui::CalcTextSize(value);
    ImGui::GetWindowDrawList()->AddText(
      ImVec2(0.5f * (lo.x + hi.x - size.x), 0.5f * (lo.y + hi.y - size.y)),
      ImGui::GetColorU32(ImGuiCol_Text),
      value
    );
    ImGui::SetItemTooltip("%.3f %s in [%.3f, %.3f]", j.q, unit, j.lo, j.hi);
}

/* Each robot a collapsible row (name, active control mode, manipulability w), its joints under
 * it without the robot's prefix: a bar between the limits, red within joint_limit_margin of
 * one, or the plain value for a joint without limits. */
static void robots_gui(WindowState *ws)
{
    if (ws->robots.empty()) {
        ImGui::TextUnformatted("No robots.");
        return;
    }
    const mjModel *m = ws->model.get();
    if (ImGui::BeginTable("##robots", 2, ImGuiTableFlags_SizingStretchProp)) {
        for (std::size_t i = 0; i < ws->robots.size(); ++i) {
            const RobotStatus &r = ws->robots[i];
            ImGui::PushID(static_cast<int>(i));
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const bool open = ImGui::TreeNodeEx(
              "##robot",
              ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth,
              "%s",
              r.label.c_str()
            );
            ImGui::TableNextColumn();
            const char    *mode = r.mode == CtrlMode::POSITION   ? "Position"
                                  : r.mode == CtrlMode::VELOCITY ? "Velocity"
                                                                 : "Torque";
            Manipulability mp;
            const bool     has_w = robot_manipulability(m, ws->data.get(), r, &mp);
            ImGui::Text("%s%s", mode, r.paused ? " (paused)" : "");
            if (has_w) {
                ImGui::SameLine();
                ImGui::TextDisabled("w %.3f", mp.w);
                ImGui::SetItemTooltip(
                  "%s",
                  "Manipulability sqrt(det(Jv Jv^T)) of the TCP's linear velocity; 0 at a "
                  "singularity"
                );
            }
            if (open) {
                for (const JointStatus &j : r.joints) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    const std::string_view name = j.name;
                    const std::string_view shown =
                      name.rfind(r.prefix, 0) == 0 ? name.substr(r.prefix.size()) : name;
                    ImGui::Indent();
                    ImGui::TextUnformatted(shown.data(), shown.data() + shown.size());
                    ImGui::Unindent();
                    ImGui::TableNextColumn();
                    const bool slide =
                      j.id >= 0 && j.id < m->njnt && m->jnt_type[j.id] == mjJNT_SLIDE;
                    const char *unit = slide ? "m" : "rad";
                    if (std::isfinite(j.lo) && std::isfinite(j.hi) && j.hi > j.lo) {
                        joint_bar(j, ws->joint_limit_margin, unit);
                    } else {
                        ImGui::TextDisabled("%.3f %s", j.q, unit);
                        ImGui::SetItemTooltip("%s", "No limits");
                    }
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::Checkbox("Mark joints near a limit", &ws->joint_limit_highlight);
    ImGui::SetItemTooltip(
      "%s", "A red sphere on the joint; window and camera views only, never recorded"
    );
    ImGui::Checkbox("Manipulability ellipsoid", &ws->manipulability_ellipsoid);
    ImGui::SetItemTooltip(
      "%s", "TCP speeds unit joint speeds reach: flat or thin near a singularity"
    );
    if (ws->manipulability_ellipsoid) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::DragFloat("##scale", &ws->manipulability_scale, 0.01f, 0.01f, 5.0f, "%.2f m");
        ImGui::SetItemTooltip("%s", "Drawn size of 1 m/s per rad/s");
    }
}

// The pre-4.0 viewer's Perturb section: what a plain left drag on the selected body applies.
static void perturb_gui(WindowState *ws)
{
    ImGui::TextUnformatted("Left drag");
    ImGui::SameLine();
    ImGui::RadioButton("None", &ws->drag_mode, WindowState::kDragNone);
    ImGui::SameLine();
    ImGui::RadioButton("Force", &ws->drag_mode, WindowState::kDragForce);
    ImGui::SameLine();
    ImGui::RadioButton("Torque", &ws->drag_mode, WindowState::kDragTorque);
    ImGui::TextDisabled(
      ws->pert.select > 0 ? "Shift: in the horizontal plane. Ctrl + drag works in any mode."
                          : "Double-click a body to select it."
    );
}

static void equality_gui(WindowState *ws, Commands *cmd)
{
    const mjModel *m = ws->model.get();
    if (m->neq == 0) {
        ImGui::TextUnformatted("No equality constraints.");
        return;
    }
    for (int i = 0; i < m->neq; ++i) {
        const char       *name   = mj_id2name(m, mjOBJ_EQUALITY, i);
        const std::string label  = name ? std::string(name) : "equality " + std::to_string(i);
        bool              active = ws->data->eq_active[i] != 0;
        if (ImGui::Checkbox((label + "##eq_" + std::to_string(i)).c_str(), &active))
            cmd->eq_active.emplace_back(i, active ? 1 : 0);
    }
}

static void recording_gui(WindowState *ws, Commands *cmd)
{
    const mjModel *m = ws->model.get();
    ImGui::BeginDisabled(ws->recording);
    ImGui::InputText("File", ws->record_path, sizeof(ws->record_path));
    std::string camera = ws->record_camera == kRecordCurrent    ? "Current"
                         : ws->record_camera == kRecordFree     ? "Free"
                         : ws->record_camera == kRecordTracking ? "Tracking"
                                                                : "";
    if (ws->record_camera >= kRecordFixed) {
        const char *name = mj_id2name(m, mjOBJ_CAMERA, ws->record_camera - kRecordFixed);
        camera = name ? name : "camera " + std::to_string(ws->record_camera - kRecordFixed);
    }
    if (ImGui::BeginCombo("Camera", camera.c_str())) {
        for (int c = 0; c < kRecordFixed + m->ncam; ++c) {
            std::string label = c == kRecordCurrent    ? "Current"
                                : c == kRecordFree     ? "Free"
                                : c == kRecordTracking ? "Tracking"
                                                       : "";
            if (c >= kRecordFixed) {
                const char *name = mj_id2name(m, mjOBJ_CAMERA, c - kRecordFixed);
                label            = name ? name : "camera " + std::to_string(c - kRecordFixed);
            }
            if (ImGui::Selectable(
                  (label + "##rec_cam_" + std::to_string(c)).c_str(), c == ws->record_camera
                ))
                ws->record_camera = c;
        }
        ImGui::EndCombo();
    }
    ImGui::Combo(
      "Resolution",
      &ws->record_resolution,
      kRecordResolutionNames,
      (int)std::size(kRecordResolutionNames)
    );
    ImGui::InputInt("FPS", &ws->record_fps);
    ws->record_fps = std::clamp(ws->record_fps, 1, 240);
    ImGui::EndDisabled();

    if (ImGui::Button(ws->recording ? "Stop recording" : "Start recording")) {
        RecordRequest req;
        req.start      = !ws->recording;
        req.path       = ws->record_path;
        req.camera     = ws->record_camera;
        req.resolution = ws->record_resolution;
        req.fps        = ws->record_fps;
        cmd->record    = req;
    }
    if (!ws->record_status.empty()) ImGui::TextUnformatted(ws->record_status.c_str());
}

// Studio's App::BuildGui, less the windows that edit or load a model the Env owns.
static void build_gui(WindowState *ws, Commands *cmd, const Gfx &gfx)
{
    if (!ws->model) return;
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    if (ws->full_screen) {
        if (ws->frame_pending > 0)
            frame_scene(ws->model.get(), ws->data.get(), &ws->cam, display.x, display.y, display.y);
        ws->frame_pending = 0;
        return;
    }
    ImGui::GetIO().FontGlobalScale = ws->font_scale;

    /* The dock splitters keep each side at least the style's WindowMinSize wide, read while the
     * dock space is laid out: raised there alone, so the panels never get narrower than their
     * widgets while popups and floating windows keep the usual minimum. The splitters apply it
     * only while dragged, so a narrower side, from a settings file, is widened here first. */
    const float panel_min = ws->min_panel_width * ImGui::GetFontSize();
    for (const char *name : { "Options", "Inspector" }) {
        const ImGuiWindow *window = ImGui::FindWindowByName(name);
        if (!window || !window->DockNode || window->DockNode->Size.x >= panel_min) continue;
        ImGui::DockBuilderSetNodeSize(
          window->DockNode->ID, { panel_min, window->DockNode->Size.y }
        );
        window->DockNode->WantLockSizeOnce = true;
    }
    ImGui::PushStyleVar(
      ImGuiStyleVar_WindowMinSize, ImVec2(panel_min, ImGui::GetStyle().WindowMinSize.y)
    );
    const ImVec4 workspace_rect = studio::ConfigureDockingLayout(ws->toolbar, ws->status_bar);
    ImGui::PopStyleVar();
    mjModel *m = ws->model.get();
    mjData  *d = ws->data.get();
    if (ws->frame_pending > 0) {
        // workspace_rect is the dock's central node, the part the panels leave free. Its size is
        // settled only frames after the layout is built, so the camera is fitted until then.
        frame_scene(m, d, &ws->cam, workspace_rect.z, workspace_rect.w, display.y);
        --ws->frame_pending;
    }

    main_menu_gui(ws, cmd, gfx);

    if (ws->toolbar) {
        if (ImGui::Begin("ToolBar")) toolbar_gui(ws, cmd);
        ImGui::End();
    }
    if (ws->status_bar) {
        studio::ScopedStyle style;
        style.Var(ImGuiStyleVar_CellPadding, ImVec2(0, 0));
        style.Var(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
        style.Var(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        if (ImGui::Begin("StatusBar")) status_bar_gui(ws);
        ImGui::End();
    }

    const float           min_width = studio::GetExpectedLabelWidth();
    const ImGuiChildFlags child_flags =
      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysAutoResize;
    const ImGuiTreeNodeFlags node_flags =
      ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Framed;
    const auto section = [&](const char *id, const char *title, auto &&body) {
        ImGui::BeginChild(id, { 0, 0 }, child_flags);
        if (studio::SectionHeader(title, node_flags, 0.65f)) {
            body();
            ImGui::TreePop();
        }
        ImGui::EndChild();
    };

    // Studio's order (App::ModelOptionsGui), with the wrapper's Recording last.
    if (ws->options_panel) {
        // The scrollbar's room is always kept, so expanding a section never narrows the panel.
        if (ImGui::Begin("Options", &ws->options_panel, ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
            simulation_gui(ws, cmd);
            section("RobotsGui", "Robots", [&] { robots_gui(ws); });
            section("PhysicsGui", "Physics", [&] {
                m->opt               = ws->env_opt;
                const mjOption shown = m->opt;
                studio::PhysicsGui(m, nullptr, min_width);
                if (std::memcmp(&shown, &m->opt, sizeof(mjOption)) != 0) cmd->opt = m->opt;
            });
            section("RenderingGui", "Rendering", [&] { rendering_gui(ws, min_width); });
            section("GroupsGui", "Visibility Groups", [&] {
                studio::GroupsGui(m, &ws->opt, min_width);
            });
            section("VisualizationGui", "Visualization", [&] {
                studio::VisualizationGui(m, &ws->opt, &ws->cam, min_width);
            });
            section("FramesGui", "Frames", [&] { frames_gui(ws); });
            section("PerturbGui", "Perturb", [&] { perturb_gui(ws); });
            section("RecordingGui", "Recording", [&] { recording_gui(ws, cmd); });
        }
        ImGui::End();
    }

    if (ws->inspector_panel) {
        /* A body picked in the scene brings its properties to the front: its tab is selected in
         * the dock's tab bar, since a click in the scene leaves no window to take focus. */
        if (ws->focus_explorer) {
            const ImGuiWindow *explorer = ImGui::FindWindowByName("Explorer");
            if (explorer && explorer->DockNode && explorer->DockNode->TabBar)
                explorer->DockNode->TabBar->NextSelectedTabId = explorer->TabId;
            ws->focus_explorer = false;
        }
        if (ImGui::Begin("Explorer", &ws->inspector_panel)) explorer_gui(ws);
        ImGui::End();

        // Studio's App::DataInspectorGui, with the wrapper's Equality section.
        if (ImGui::Begin(
              "Inspector", &ws->inspector_panel, ImGuiWindowFlags_AlwaysVerticalScrollbar
            )) {
            section("JointsGui", "Joints", [&] { studio::JointsGui(m, d, &ws->opt); });
            section("ControlsGui", "Controls", [&] {
                float scale = 0, rate = 0;
                ws->step_control.GetNoiseParameters(scale, rate);
                studio::NoiseGui(&ws->step_control);
                float new_scale = 0, new_rate = 0;
                ws->step_control.GetNoiseParameters(new_scale, new_rate);
                if (new_scale != scale || new_rate != rate) cmd->noise = { new_scale, new_rate };
                ImGui::Separator();
                const std::vector<mjtNum> shown(d->ctrl, d->ctrl + m->nu);
                studio::ControlsGui(m, d, &ws->opt);
                for (int a = 0; a < m->nu; ++a)
                    if (d->ctrl[a] != shown[a]) cmd->ctrl.emplace_back(a, d->ctrl[a]);
            });
            section("SensorGui", "Sensor", [&] { studio::SensorGui(m, d); });
            section("WatchGui", "Watch", [&] {
                studio::WatchGui(m, d, ws->watch_field, sizeof(ws->watch_field), ws->watch_index);
            });
            section("StateGui", "State", [&] {
                // StateGui writes an edited state into the data: the Env takes it over.
                const int           sig = ws->state_sig;
                std::vector<mjtNum> before(mj_stateSize(m, sig));
                mj_getState(m, d, before.data(), sig);
                studio::StateGui(m, d, ws->state, ws->state_sig, min_width);
                std::vector<mjtNum> after(before.size());
                mj_getState(m, d, after.data(), sig);
                if (sig == ws->state_sig && after != before) cmd->state = { sig, after };
            });
            section("EqualityGui", "Equality", [&] { equality_gui(ws, cmd); });
        }
        ImGui::End();
    }

    if (ws->profiler_open) {
        if (ImGui::Begin(
              "Profiler",
              &ws->profiler_open,
              ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse
            ))
            studio::ProfilerGui(m, d, &ws->profiler, ws->profiler_show_iter);
        ImGui::End();
    }

    if (ws->console) {
        if (ImGui::Begin("Console", &ws->console)) {
            studio::Console::Get().WindowGui();
        } else if (ImGui::IsWindowCollapsed()) {
            ImGui::SetWindowCollapsed(false);
            ws->console = false;
        }
        ImGui::End();
    }

    const float view_min = ws->min_panel_width * ImGui::GetFontSize();
    for (auto view = ws->camera_views.begin(); view != ws->camera_views.end();) {
        bool              open = true;
        const char       *name = mj_id2name(m, mjOBJ_CAMERA, view->pip.camera);
        const std::string title =
          std::string(name ? name : "Camera " + std::to_string(view->pip.camera)) + "###camera_view"
          + std::to_string(view->id);
        dock_beside_inspector();
        // Floating once pulled out of the dock, where the constraint applies (docked, the panel
        // width does).
        ImGui::SetNextWindowSizeConstraints({ view_min, view_min }, { FLT_MAX, FLT_MAX });
        if (ImGui::Begin(title.c_str(), &open))
            camera_view_gui(ws, gfx, &view->pip, gfx.window->GetAspectRatio());
        ImGui::End();
        if (open) {
            ++view;
            continue;
        }
        if (const auto size = ws->pip_sizes.find(view->pip.texture); size != ws->pip_sizes.end()) {
            ws->retired_textures.push_back(view->pip.texture);
            ws->pip_sizes.erase(size);
        }
        view = ws->camera_views.erase(view);
    }

    if (ws->help) {
        studio::ScopedStyle style;
        style.Var(ImGuiStyleVar_Alpha, 0.8f);
        ImGui::SetNextWindowPos(ImVec2(workspace_rect.x, workspace_rect.y), ImGuiCond_Appearing);
        ImGui::SetNextWindowSize(ImVec2(0, 0), ImGuiCond_Appearing);
        if (ImGui::Begin("Help", &ws->help, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(2.0f, 5.0f));
            help_gui();
            ImGui::PopStyleVar();
        }
        ImGui::End();
    }

    if (ws->info) {
        const float scale = ImGui::GetWindowDpiScale();
        if (studio::BeginOverlay(
              "Info", studio::OverlayPos::kBottomLeft, workspace_rect, 180.0f * scale
            ))
            studio::InfoGui(m, d, ws->paused(), (float)ws->fps, ws->gpu_ms, ws->timer_prev.data());
        studio::EndOverlay();
    }
    std::copy_n(d->timer, mjNTIMER, ws->timer_prev.begin());

    if (ws->info && ws->paused())
        studio::TextOverlay(
          "Pause", studio::OverlayPos::kTop, workspace_rect, "PAUSE", ImVec4(0, 0, 0, 0), 3.0f
        );

    // The speed is measured on the Env's thread, which does the stepping.
    if (!ws->paused()) {
        const float desired  = ws->step_control.GetSpeed();
        const float measured = static_cast<float>(ws->speed_measured);
        if (desired != 100.0f || std::abs(measured - desired) > 0.1f * desired) {
            char       label[30];
            const bool misaligned = std::abs(measured - desired) > 0.1f * desired;
            if (misaligned)
                std::snprintf(label, sizeof(label), "%g%% (%-4.1f%%)", desired, measured);
            else
                std::snprintf(label, sizeof(label), "%g%%", desired);
            studio::TextOverlay(
              "Realtime",
              studio::OverlayPos::kTopLeft,
              workspace_rect,
              label,
              ImVec4(0, 0, 0, 0),
              2.0f
            );
        }
    }

    if (ws->view_gizmo)
        studio::ViewGizmoGui(m, d, ws->cam, ws->camera_idx, ws->gizmo_animation, workspace_rect);

    if (ImGui::BeginPopup("SceneContextMenu")) {
        if (ImGui::MenuItem("Deselect")) {
            ws->element = nullptr;
            mjv_defaultPerturb(&ws->pert);
        }
        ImGui::EndPopup();
    }

    file_dialog_gui(ws, cmd, gfx);

    if (ws->style_editor) {
        dock_beside_inspector();
        if (ImGui::Begin("Style Editor", &ws->style_editor)) style_editor_gui(ws);
        ImGui::End();
    }

    if (mjrfScene *scene = gfx.renderer->GetScene("scene")) {
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("Plugins")) {
                if (ImGui::MenuItem("Filament", "", ws->filament_editor))
                    ws->filament_editor = !ws->filament_editor;
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
        if (ws->filament_editor) {
            if (ImGui::Begin("Filament", &ws->filament_editor)) mujoco::DrawSceneEditor(scene);
            ImGui::End();
        }
    }
    studio::ForEachPlugin<studio::GuiPlugin>([](studio::GuiPlugin *plugin) {
        if (!plugin->update) return;
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("Plugins")) {
                if (ImGui::MenuItem(plugin->name, "", plugin->active))
                    plugin->active = !plugin->active;
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
        if (plugin->active) {
            if (ImGui::Begin(plugin->name, &plugin->active)) plugin->update(plugin);
            ImGui::End();
        }
    });

    ImGuiIO &io = ImGui::GetIO();
    if (io.WantSaveIniSettings) {
        studio::SaveSettings(ws->theme, ws->ini_path);
        io.WantSaveIniSettings = false;
    }
}

// The ImGui layer: an orthographic camera over the window in pixels.
static mujoco::Renderer::View ux_view(mujoco::Renderer *renderer, int width, int height)
{
    mujoco::Renderer::View view;
    view.scene                 = renderer->GetSceneHandle("ux");
    view.shadows               = false;
    view.reflections           = false;
    view.post_processing       = false;
    view.camera.orthographic   = true;
    view.camera.pos[2]         = 1.0f;
    view.camera.forward[2]     = -1.0f;
    view.camera.up[1]          = 1.0f;
    view.camera.frustum_far    = 1.0f;
    view.camera.frustum_center = width / 2.0f;
    view.camera.frustum_width  = width / 2.0f;
    view.camera.frustum_bottom = height;
    return view;
}

/* A frame drawn as MuJoCo draws frames (addFrame in engine_vis_visualize.c): a cylinder per
 * axis, x red, y green, z blue, as long and wide as the model's vis.scale.framelength and
 * framewidth times its mean size. A selected joint's frame sits at its anchor, in its body's
 * orientation; an element without a pose in the model adds nothing. */
static void
  add_frame(const mjModel *m, const mjtNum *pos, const mjtNum *rot, std::vector<mjvGeom> *geoms)
{
    const mjtNum length = m->vis.scale.framelength * m->stat.meansize;
    const mjtNum width  = m->vis.scale.framewidth * m->stat.meansize;
    for (int axis = 0; axis < 3; ++axis) {
        mjtNum unit[3] = { 0, 0, 0 }, vec[3], to[3];
        unit[axis]     = length;
        mju_mulMatVec3(vec, rot, unit);
        mju_add3(to, pos, vec);
        mjvGeom &g = geoms->emplace_back();
        mjv_initGeom(&g, mjGEOM_NONE, nullptr, nullptr, nullptr, nullptr);
        mjv_connector(&g, mjGEOM_CYLINDER, width, pos, to);
        for (int k = 0; k < 3; ++k) g.rgba[k] = axis == k ? 0.9f : 0.0f;
        g.rgba[3] = 1.0f;
    }
}

// The frames checked in Frames, and that of what the Explorer has selected.
static void add_frames(const WindowState *ws, std::vector<mjvGeom> *geoms)
{
    const mjModel *m = ws->model.get();
    const mjData  *d = ws->data.get();
    if (!ws->shown_frames.empty()) {
        for (int b = 0; b < m->nbody; ++b)
            if (ws->shown_frames.count({ mjOBJ_BODY, frame_label(m, mjOBJ_BODY, b) }))
                add_frame(m, d->xpos + 3 * b, d->xmat + 9 * b, geoms);
        for (int s = 0; s < m->nsite; ++s)
            if (ws->shown_frames.count({ mjOBJ_SITE, frame_label(m, mjOBJ_SITE, s) }))
                add_frame(m, d->site_xpos + 3 * s, d->site_xmat + 9 * s, geoms);
    }
    const mjtNum *pos = nullptr, *rot = nullptr;
    if (ws->element && element_world_frame(m, d, ws->element, &pos, &rot))
        add_frame(m, pos, rot, geoms);
}

/* A translucent red sphere on each robot joint within joint_limit_margin of a limit. The joint
 * sits inside its link's mesh, so the sphere is half the model's mean body size (stat.meansize)
 * across, enough to show around a link. */
static void add_limit_spheres(const WindowState *ws, std::vector<mjvGeom> *geoms)
{
    if (!ws->joint_limit_highlight) return;
    const mjModel *m       = ws->model.get();
    const mjData  *d       = ws->data.get();
    const mjtNum   size[3] = { 0.5 * m->stat.meansize, 0, 0 };
    const float    rgba[4] = { 0.9f, 0.1f, 0.1f, 0.5f };
    for (const RobotStatus &r : ws->robots)
        for (const JointStatus &j : r.joints)
            if (j.id >= 0 && j.id < m->njnt && joint_near_limit(j, ws->joint_limit_margin)) {
                mjvGeom &g = geoms->emplace_back();
                mjv_initGeom(&g, mjGEOM_SPHERE, size, d->xanchor + 3 * j.id, nullptr, rgba);
            }
}

// Each robot's manipulability ellipsoid at its TCP, radii scaled by manipulability_scale.
static void add_manipulability(const WindowState *ws, std::vector<mjvGeom> *geoms)
{
    if (!ws->manipulability_ellipsoid) return;
    const float rgba[4] = { 0.25f, 0.6f, 1.0f, 0.35f };
    for (const RobotStatus &r : ws->robots) {
        Manipulability mp;
        if (!robot_manipulability(ws->model.get(), ws->data.get(), r, &mp)) continue;
        mjtNum size[3];
        for (int k = 0; k < 3; ++k)
            size[k] = std::max<mjtNum>(ws->manipulability_scale * mp.radii[k], 1e-4);
        mjvGeom &g = geoms->emplace_back();
        mjv_initGeom(&g, mjGEOM_ELLIPSOID, size, mp.pos, mp.axes, rgba);
    }
}

static void render(WindowState *ws, const Gfx &gfx)
{
    const float   scale    = gfx.window->GetScale();
    const mjrRect viewport = { 0,
                               0,
                               static_cast<int>(gfx.window->GetWidth() * scale),
                               static_cast<int>(gfx.window->GetHeight() * scale) };
    if (ws->scene) {
        // Studio's ScenePlugins add their geoms on top of the Env's overlays.
        ws->plugin_scene.ngeom = 0;
        studio::ForEachPlugin<studio::ScenePlugin>([&](studio::ScenePlugin *plugin) {
            if (plugin->enhance_scene)
                plugin->enhance_scene(plugin, ws->model.get(), ws->data.get(), &ws->plugin_scene);
        });
        std::vector<mjvGeom> extra = ws->overlays;
        extra.insert(
          extra.end(), ws->plugin_scene.geoms, ws->plugin_scene.geoms + ws->plugin_scene.ngeom
        );
        add_frames(ws, &extra);
        add_limit_spheres(ws, &extra);
        add_manipulability(ws, &extra);
        ws->scene->Sync(
          ws->model.get(), ws->data.get(), viewport, &ws->opt, &ws->pert, &ws->cam, extra
        );
        ws->scene_geoms = std::move(extra);
    }
    gfx.bridge->Update();

    std::vector<mujoco::Renderer::View> views;
    if (ws->scene) views.push_back(scene_view(ws, gfx.renderer, ws->scene.get(), &ws->cam));
    views.push_back(ux_view(gfx.renderer, viewport.width, viewport.height));
    gfx.renderer->ResizeRenderTarget(
      mujoco::Renderer::kWindowRenderTarget, viewport.width, viewport.height
    );
    const auto target   = gfx.renderer->GetRenderTargetHandle("pixels");
    const bool headless = studio::IsHeadless(gfx.window->GetGraphicsMode());
    if (headless) {
        gfx.renderer->ResizeRenderTarget(target, viewport.width, viewport.height);
        for (auto &view : views) view.render_target = target;
    }
    gfx.renderer->Render(views);
    ws->gpu_ms = gfx.renderer->GetGpuFrameMs();

    // A texture replaced before this frame is drawn by nothing once its render has run.
    for (ImTextureID texture : ws->retired_textures) gfx.bridge->UploadImage(texture, {}, 0, 0);
    ws->retired_textures.clear();
    gfx.window->EndFrame();
    gfx.window->Present(headless ? gfx.renderer->GetPixels(target) : std::span<const std::byte>{});
}

/* Studio's Window calls mju_error, which ends the process, when SDL cannot start or open a
 * window. Make the same two calls first, so a missing display is an error open_viewer() returns. */
static std::string window_error(studio::GraphicsMode mode)
{
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0)
        return std::string("SDL could not be initialised: ") + SDL_GetError();
    // SDL falls back to drivers without a desktop (KMSDRM, offscreen); Filament then exits.
    const char            *current = SDL_GetCurrentVideoDriver();
    const std::string_view driver  = current ? current : "";
    if (driver != "x11" && driver != "wayland") {
        SDL_Quit();
        return "no X11 or Wayland display (SDL video driver '" + std::string(driver) + "')";
    }
    Uint32 flags = SDL_WINDOW_HIDDEN;
    if (studio::IsOpenGl(mode) && !studio::IsHeadless(mode)) flags |= SDL_WINDOW_OPENGL;
    SDL_Window *probe = SDL_CreateWindow("probe", 0, 0, 1, 1, flags);
    if (!probe) {
        std::string error = std::string("the window could not be created: ") + SDL_GetError();
        SDL_Quit();
        return error;
    }
    SDL_DestroyWindow(probe);
    return {};
}

static void window_main(ViewerImpl *impl)
{
    // Wayland takes no direct OpenGL window: Studio renders offscreen there and blits the frame.
    const char                *session = std::getenv("XDG_SESSION_TYPE");
    const studio::GraphicsMode mode =
      (session && std::string_view(session) == "wayland") || std::getenv("WAYLAND_DISPLAY")
        ? studio::GraphicsMode::FilamentOpenGlHeadless
        : studio::GraphicsMode::FilamentOpenGl;
    if (std::string error = window_error(mode); !error.empty()) {
        {
            const std::lock_guard<std::mutex> lock(impl->mtx);
            impl->start_error = std::move(error);
        }
        impl->closed = true;
        impl->started_cv.notify_all();
        return;
    }
    register_asset_providers();
    studio::Console::Init();
    // open_viewer() waits for this thread to start, so the caller's settings hold still here.
    const Viewer  &settings = *impl->owner;
    studio::Window window(
      impl->title, settings.window_width, settings.window_height, { .gfx_mode = mode }
    );

    mjrfContextConfig config;
    mjrf_defaultContextConfig(&config);
    config.native_window      = window.GetNativeWindowHandle();
    config.software_rendering = studio::IsSoftware(mode);
    config.graphics_api = studio::IsOpenGl(mode) || studio::IsWebGl(mode) ? mjGRAPHICS_API_OPENGL
                                                                          : mjGRAPHICS_API_VULKAN;
    mujoco::Renderer       renderer(config);
    mjrfRenderTargetConfig target;
    mjrf_defaultRenderTargetConfig(&target);
    target.color_format = mjPIXEL_FORMAT_RGB8;
    target.depth_format = mjPIXEL_FORMAT_DEPTH32F;
    renderer.CreateRenderTarget("pixels", target);
    renderer.GetScene("scene");
    mujoco::ImguiBridge bridge(renderer.GetContext(), renderer.GetScene("ux"));
    ImPlot::CreateContext();
    const Gfx gfx{ &window, &renderer, &bridge };

    // By default Studio's own settings file, so the layout and theme carry over between the two.
    WindowState ws;
    ws.ini_path      = settings.ini_path.empty() ? studio::GetDefaultIniPath() : settings.ini_path;
    ws.default_theme = settings.theme == ViewerTheme::Light     ? studio::GuiTheme::kLight
                       : settings.theme == ViewerTheme::Classic ? studio::GuiTheme::kClassic
                                                                : studio::GuiTheme::kDark;
    name_studio_fonts();
    register_style_settings(&ws.style);
    load_settings(&ws);
    ws.framing_frames           = settings.framing_frames;
    ws.min_panel_width          = settings.min_panel_width;
    ws.joint_limit_margin       = std::clamp(settings.joint_limit_margin, 0.0, 0.5);
    ws.joint_limit_highlight    = settings.joint_limit_highlight;
    ws.manipulability_ellipsoid = settings.manipulability_ellipsoid;
    ws.manipulability_scale     = static_cast<float>(settings.manipulability_scale);
    const std::size_t path_len  = std::min(settings.record_path.size(), sizeof(ws.record_path) - 1);
    settings.record_path.copy(ws.record_path, path_len);
    ws.record_path[path_len] = '\0';
    const auto *resolution   = std::ranges::find(kRecordResolutions, settings.record_resolution);
    ws.record_resolution     = resolution == std::end(kRecordResolutions)
                                 ? 0
                                 : static_cast<int>(resolution - kRecordResolutions);
    ws.record_fps            = settings.record_fps;
    mjv_defaultOption(&ws.opt);
    mjv_defaultPerturb(&ws.pert);
    mjv_makeScene(nullptr, &ws.plugin_scene, settings.plugin_max_geoms);
    for (int i = 0; i < mjNRNDFLAG; ++i) ws.render_flags[i] = mjRNDSTRING[i][1][0] == '1';
    double realtime_factor = 1.0;
    {
        const std::lock_guard<std::mutex> lock(impl->mtx);
        ws.cam = impl->camera;
        ws.step_control.SetPauseState(
          impl->paused ? PauseState::kNormalPaused : PauseState::kUnpaused
        );
        realtime_factor = impl->owner->realtime_factor;
        impl->started   = true;
    }
    impl->started_cv.notify_all();

    // The panel's speeds stop at real time; a faster or uncapped (0) factor stays until changed.
    int start_index = 0;
    for (int i = 0; i < (int)studio::kPercentRealTime.size(); ++i) {
        const double pct = std::stod(studio::kPercentRealTime[i]);
        if (std::abs(pct - realtime_factor * 100.0)
            < std::abs(std::stod(studio::kPercentRealTime[start_index]) - realtime_factor * 100.0))
            start_index = i;
    }
    studio::SetSpeedIndex(&ws.step_control, ws.speed_index, start_index);
    float sent_speed = ws.step_control.GetSpeed();

    auto last = std::chrono::steady_clock::now();
    while (!impl->stop) {
        prepare_style(&ws);
        if (window.NewFrame() == studio::Window::kQuitting) break;
        const auto   now = std::chrono::steady_clock::now();
        const double dt  = std::chrono::duration<double>(now - last).count();
        last             = now;
        if (dt > 0) ws.fps = ws.fps > 0 ? 0.9 * ws.fps + 0.1 / dt : 1.0 / dt;

        take_snapshot(impl, &ws, &renderer);
        Commands cmd;
        if (ws.model) {
            handle_mouse(&ws, window);
            handle_keys(impl, &ws, &cmd);
        }
        build_gui(&ws, &cmd, gfx);
        render(&ws, gfx);
        if (ws.quit) break;

        if (ws.step_control.GetSpeed() != sent_speed) {
            sent_speed          = ws.step_control.GetSpeed();
            cmd.realtime_factor = sent_speed / 100.0;
        }
        cmd.single = cmd.single || ws.step_control.ConsumeSingleStepRequest();
        if (ws.update_threadpool) {
            cmd.nthread          = ws.nthread;
            ws.update_threadpool = false;
        }
        // Step Back/Fwd and the scrubber moved the timeline: the Env loads that frame.
        if (ws.timeline.history_index != ws.env_history_index) {
            cmd.history_index = ws.timeline.history_index;
            ws.step_control.SetPauseState(PauseState::kNormalPaused);
        }

        const std::lock_guard<std::mutex> lock(impl->mtx);
        Commands                         &out = impl->commands;
        if (cmd.realtime_factor) out.realtime_factor = cmd.realtime_factor;
        if (cmd.reset) {
            out.reset    = true;
            out.keyframe = cmd.keyframe;
        }
        out.reload = out.reload || cmd.reload;
        out.single = out.single || cmd.single;
        if (cmd.history_index) out.history_index = cmd.history_index;
        if (cmd.save_keyframe) out.save_keyframe = cmd.save_keyframe;
        if (cmd.nthread) out.nthread = cmd.nthread;
        if (cmd.noise) out.noise = cmd.noise;
        if (cmd.state) out.state = std::move(cmd.state);
        out.ctrl.insert(out.ctrl.end(), cmd.ctrl.begin(), cmd.ctrl.end());
        out.eq_active.insert(out.eq_active.end(), cmd.eq_active.begin(), cmd.eq_active.end());
        if (cmd.opt) out.opt = cmd.opt;
        if (cmd.record) out.record = cmd.record;
        if (cmd.save_xml) out.save_xml = cmd.save_xml;
        impl->paused  = ws.paused();
        impl->perturb = ws.pert;
        impl->camera  = ws.cam;
        impl->fps     = ws.fps;
    }

    // Filament objects belong to the thread that made them, so they go here, scene first. The
    // ImGui bridge frees its textures with the renderables that draw them, as in Studio.
    ws.scene.reset();
    ws.offscreen.scene.reset();
    ws.offscreen.renderer.reset();
    ws.data.reset();
    mjv_freeScene(&ws.plugin_scene);
    // As Studio's App on exit: ImGui's autosave runs on a timer, so the last changes would be lost.
    studio::SaveSettings(ws.theme, ws.ini_path);
    ImGui::RemoveSettingsHandler("MjKdl"); // it points into ws, which goes before ImGui's context
    ImPlot::DestroyContext();
    impl->closed = true;
}

// Public API

Viewer::~Viewer() { cleanup(this); }

Status open_viewer(Viewer *v, Env *env, const char *title)
{
    if (!v || !env || !env->model) MJ_FAIL("open_viewer: null viewer or empty env");
    if (v->_impl) MJ_FAIL("open_viewer: the viewer is already open");
    if (!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY"))
        MJ_FAIL("open_viewer: no display (neither DISPLAY nor WAYLAND_DISPLAY is set)");

    auto *impl         = new ViewerImpl();
    impl->owner        = v;
    impl->env          = env;
    impl->title        = title ? title : "MuJoCo";
    impl->camera       = v->cam;
    impl->paused       = v->paused;
    impl->speed_window = v->speed_window;
    mjv_defaultPerturb(&impl->perturb);
    studio::ResetHistory(impl->history, impl->timeline, env->model, env->data);
    impl->history_stale = false;
    publish(impl, env);

    EnvHooks &hooks  = env->hooks[impl];
    hooks.on_step    = [impl](Env *e, bool *integrate) { return on_step(impl, e, integrate); };
    hooks.on_rebuild = [impl](Env *) {
        // Camera, actuator and body ids move with a recompile, so none of them carries over.
        if (impl->recording) {
            MJ_LOG_WARN("scene rebuilt, recording stopped");
            stop_recording(impl, "scene rebuilt, recording stopped");
        }
        impl->ctrl_override.clear();
        impl->pert_body     = -1;
        impl->model_stale   = true;
        impl->history_stale = true;
        impl->stepped       = false;
    };
    hooks.on_reset = [impl](Env *) {
        impl->ctrl_override.clear();
        impl->owner->_tick_t = {};
        impl->history_stale  = true;
    };
    hooks.on_close = [impl](Env *) {
        if (impl->recording) stop_recording(impl, "recording stopped");
        close_window(impl);
        impl->env = nullptr;
    };

    impl->thread = std::thread(window_main, impl);
    std::string error;
    {
        std::unique_lock<std::mutex> lock(impl->mtx);
        impl->started_cv.wait(lock, [impl] { return impl->started || impl->closed; });
        error = impl->start_error;
    }
    if (!error.empty()) {
        impl->thread.join();
        env->hooks.erase(impl);
        delete impl;
        MJ_FAIL("open_viewer: " << error);
    }
    v->_impl = impl;
    return {};
}

void cleanup(Viewer *v)
{
    if (!v || !v->_impl) return;
    auto *impl = static_cast<ViewerImpl *>(v->_impl);
    if (impl->recording) stop_recording(impl, "recording stopped");
    close_window(impl);
    if (impl->env) impl->env->hooks.erase(impl);
    delete impl;
    v->_impl = nullptr;
}

void pace_realtime(Viewer *v)
{
    using Clock = std::chrono::steady_clock;
    using Dur   = std::chrono::duration<double>;
    if (!v || !v->_impl) return;
    const auto *impl = static_cast<ViewerImpl *>(v->_impl);
    if (!impl->env || !impl->env->model) return;
    const double wall_per_step =
      (v->realtime_factor > 0.0) ? impl->env->model->opt.timestep / v->realtime_factor : 0.0;
    const auto now = Clock::now();
    if (wall_per_step > 0.0 && v->_tick_t.time_since_epoch().count() != 0) {
        // In the clock's own duration, so the deadline carries forward without a lossy cast.
        const auto period = std::chrono::duration_cast<Clock::duration>(Dur(wall_per_step));
        const auto next   = v->_tick_t + period;
        if (now < next) {
            std::this_thread::sleep_until(next);
            /* Carry the deadline rather than restarting from the wake time: sleep_until
             * overshoots by tens of microseconds, and measuring the next period from the
             * moment we woke would fold that into every step -- roughly 1% at 2 ms and 2% at
             * 1 ms. */
            v->_tick_t = next;
            return;
        }
    }
    // Already past the deadline: resynchronise, so a loop back from a stall does not burst.
    v->_tick_t = now;
}

bool is_running(const Viewer *v)
{
    if (!v || !v->_impl) return false;
    const auto *impl = static_cast<const ViewerImpl *>(v->_impl);
    return impl->env && !impl->closed;
}

bool key_pressed(const Viewer *v, int imgui_key)
{
    const int k = key_index(imgui_key);
    if (!v || !v->_impl || k < 0) return false;
    return static_cast<const ViewerImpl *>(v->_impl)->keys[k].load(std::memory_order_relaxed);
}

void capture_key(Viewer *v, int imgui_key, bool capture)
{
    const int k = key_index(imgui_key);
    if (!v || !v->_impl || k < 0) return;
    static_cast<ViewerImpl *>(v->_impl)->captured[k].store(capture, std::memory_order_relaxed);
}

bool use_camera(Viewer *v, const mjModel *model, const char *name)
{
    if (!v || !model) return false;
    mjvCamera              cam      = v->cam;
    const std::string_view cam_name = name ? name : "";
    if (cam_name.empty()) {
        mjv_defaultFreeCamera(model, &cam);
    } else {
        const int id = mj_name2id(model, mjOBJ_CAMERA, name);
        if (id < 0) return false;
        cam.type       = mjCAMERA_FIXED;
        cam.fixedcamid = id;
    }
    v->cam = cam;
    if (v->_impl) {
        auto                             *impl = static_cast<ViewerImpl *>(v->_impl);
        const std::lock_guard<std::mutex> lock(impl->mtx);
        impl->camera_request = cam;
    }
    return true;
}

void set_free_camera(
  Viewer                      *v,
  double                       distance,
  double                       azimuth,
  double                       elevation,
  const std::array<double, 3> &lookat
)
{
    if (!v) return;
    v->cam.type       = mjCAMERA_FREE;
    v->cam.fixedcamid = -1;
    v->cam.distance   = distance;
    v->cam.azimuth    = azimuth;
    v->cam.elevation  = elevation;
    v->cam.lookat[0]  = lookat[0];
    v->cam.lookat[1]  = lookat[1];
    v->cam.lookat[2]  = lookat[2];
    if (v->_impl) {
        auto                             *impl = static_cast<ViewerImpl *>(v->_impl);
        const std::lock_guard<std::mutex> lock(impl->mtx);
        impl->camera_request = v->cam;
    }
}

} // namespace mj_kdl
