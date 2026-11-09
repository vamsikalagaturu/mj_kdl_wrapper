/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#include "mj_kdl_wrapper/viewer.hpp"
#include "mj_kdl_wrapper/render.hpp"

#include <SDL.h>
#include <imgui.h>
#include <implot.h>
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
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace mj_kdl {

void    register_asset_providers();            // defined in render.cpp
mjSpec *copy_model_spec(const mjModel *model); // defined in mj_kdl_wrapper.cpp
Status  reload_scene(Env *env);                // defined in mj_kdl_wrapper.cpp

namespace studio = mujoco::studio;
using PauseState = studio::StepControl::PauseState;

static constexpr int kWindowWidth  = 1600;
static constexpr int kWindowHeight = 900;

static constexpr VideoResolution kRecordResolutions[]     = { VideoResolution::R360p,
                                                              VideoResolution::R480p,
                                                              VideoResolution::R720p,
                                                              VideoResolution::R1080p };
static constexpr const char     *kRecordResolutionNames[] = { "360p", "480p", "720p", "1080p" };
static constexpr int             kRecordDefaultResolution = 2;
static constexpr int             kRecordDefaultFps        = 30;

// Recording cameras before the model's own: 0 the window's, 1 free, 2 tracking the selection.
enum RecordCamera { kRecordCurrent = 0, kRecordFree = 1, kRecordTracking = 2, kRecordFixed = 3 };

// ImGui sizes a new dock layout over its first frames; the start camera is fitted through them.
static constexpr int kFramingFrames = 5;

// The measured speed is the sim time over the wall time of this long a window.
static constexpr double kSpeedWindowS = 0.5;

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
    int         resolution = kRecordDefaultResolution;
    int         fps        = kRecordDefaultFps;
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

// The Env as step() last published it, for the window's thread.
struct Snapshot
{
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
    const char  *path = req.path.empty() ? "recording.mp4" : req.path.c_str();
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
    if (wall < kSpeedWindowS) return;
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
        cmd              = std::move(impl->commands);
        impl->commands   = Commands{};
        paused           = impl->paused;
        pert             = impl->perturb;
        cam              = impl->camera;
        impl->owner->fps = impl->fps;
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
        if (dragged >= 0) mjv_applyPerturbForce(m, d, &pert);
        inject_noise(impl, m, d);
        studio::ForEachPlugin<studio::ModelPlugin>([&](studio::ModelPlugin *plugin) {
            if (plugin->pre_step) plugin->pre_step(plugin, m, d);
        });
        impl->stepped = true;
    } else {
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

// The window's Filament objects, which the GUI renders off-screen with (screenshot, PiP).
struct Gfx
{
    studio::Window      *window   = nullptr;
    mujoco::Renderer    *renderer = nullptr;
    mujoco::ImguiBridge *bridge   = nullptr;
};

// What the window's thread keeps between frames: Studio's App state, less the simulation.
struct WindowState
{
    std::shared_ptr<mjModel>            model;
    std::uint64_t                       model_gen = 0;
    DataPtr                             data;
    std::unique_ptr<studio::ModelScene> scene;
    std::vector<mjvGeom>                overlays;
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

    int              frame_pending = 0; // frames left to fit the start camera to the layout
    std::string      ini_path;
    studio::GuiTheme theme      = studio::GuiTheme::kDark;
    float            font_scale = 1.0f;
    bool             quit       = false;

    // App::UiTempState's windows.
    bool help               = false;
    bool info               = false;
    bool profiler_open      = false;
    bool profiler_show_iter = false;
    bool picture_in_picture = false;
    bool options_panel      = true;
    bool inspector_panel    = true;
    bool toolbar            = false;
    bool view_gizmo         = true;
    bool status_bar         = false;
    bool full_screen        = false;
    bool console            = false;
    bool style_editor       = false;
    bool filament_editor    = false;

    std::vector<studio::PipState> pips;
    char                          watch_field[1000] = "qpos";
    int                           watch_index       = 0;
    std::vector<mjtNum>           state;
    int                           state_sig = 0;

    std::shared_ptr<mjSpec> spec;
    mjsElement             *element        = nullptr; // selected in the Explorer
    float                   explorer_split = -1.0f;
    int                     element_mode   = 0; // 0 spec, 1 model, 2 data

    enum FileDialog { kNoDialog, kSaveXml, kSaveMjb, kSaveScreenshot, kPrintModel, kPrintData };
    FileDialog file_dialog    = kNoDialog;
    char       filename[1024] = "";

    char        record_path[1024] = "recording.mp4";
    int         record_camera     = kRecordCurrent;
    int         record_resolution = kRecordDefaultResolution;
    int         record_fps        = kRecordDefaultFps;
    bool        recording         = false;
    std::string record_status;
    double      fps    = 0.0;
    float       gpu_ms = 0.0f;

    bool paused() const { return step_control.GetPauseState() != PauseState::kUnpaused; }
};

// Wayland takes no direct OpenGL window: Studio renders offscreen there and blits the frame.
static studio::GraphicsMode graphics_mode()
{
    const char *session = std::getenv("XDG_SESSION_TYPE");
    if ((session && std::string_view(session) == "wayland") || std::getenv("WAYLAND_DISPLAY"))
        return studio::GraphicsMode::FilamentOpenGlHeadless;
    return studio::GraphicsMode::FilamentOpenGl;
}

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
            ws->frame_pending = kFramingFrames;
    }
    if (got_data) ws->profiler.Update(ws->model.get(), ws->data.get());
}

static void handle_mouse(WindowState *ws, const studio::Window &window)
{
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;
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

    // Ctrl + drag pushes the selected body, as in Studio.
    if (dragging && io.KeyCtrl && ws->pert.select > 0) {
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

    studio::MoveCameraWithMouse(
      m, &ws->cam, ws->camera_idx == studio::kFreeCameraIdx, studio::ReadImGuiCameraInput()
    );

    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        const studio::PickResult picked =
          studio::Pick(m, d, &ws->cam, mouse_x, mouse_y, window.GetAspectRatio(), &ws->opt);
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
    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Right)) {
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

static mujoco::Renderer::View
  scene_view(const WindowState *ws, mujoco::Renderer *renderer, const mjvCamera *camera)
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
        view.headlight->intensity = ws->scene->GetHeadlightIntensity();
    }
    return view;
}

// Studio's App::RenderToTexture: the scene through `camera`, into the "tmp" target.
static std::span<const std::byte>
  render_to_texture(WindowState *ws, const Gfx &gfx, mjvCamera *camera, int width, int height)
{
    const auto target = gfx.renderer->GetRenderTargetHandle("tmp");
    gfx.renderer->ResizeRenderTarget(target, width, height);
    mujoco::Renderer::View views[1] = { scene_view(ws, gfx.renderer, camera) };
    views[0].render_target          = target;
    gfx.renderer->Render(views);
    return gfx.renderer->GetPixels(target);
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
}

// The Explorer as Studio draws it (App::SpecExplorerGui), on a read-only copy of the spec.
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
        studio::SpecTreeGui(&element, ws->spec.get());
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
            ImGui::Text("%s", mju_type2Str(ws->element->elemtype));
            ImGui::SameLine();
            ImGui::Text("(%d)", mjs_getId(ws->element));
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 100.0f);
            const char *labels[]   = { "S", "M", "D" };
            const char *tooltips[] = { "Spec", "Model", "Data" };
            for (int mode = 0; mode < 3; ++mode) {
                if (mode > 0) ImGui::SameLine();
                studio::ScopedStyle style;
                style.Var(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
                if (ws->element_mode == mode) style.Color(ImGuiCol_Button, ImGuiCol_ButtonActive);
                if (ImGui::Button(labels[mode], ImVec2(24.0f, 20.0f))) ws->element_mode = mode;
                ImGui::SetItemTooltip("%s", tooltips[mode]);
            }
            ImGui::Separator();
            if (ws->element_mode == 0)
                studio::ElementSpecGui(element);
            else if (ws->element_mode == 1)
                studio::ElementModelGui(ws->model.get(), ws->element);
            else
                studio::ElementDataGui(ws->data.get(), ws->element);
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
            if (ImGui::MenuItem("Save Config")) studio::SaveSettings(ws->theme, ws->ini_path);
            if (ImGui::MenuItem("Reset Config")) {
                studio::ResetConfig(ws->ini_path);
                ws->theme = studio::LoadSettings(ws->ini_path, studio::GuiTheme::kDark);
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
            if (ImGui::MenuItem("Picture-in-Picture", nullptr, ws->picture_in_picture))
                ws->picture_in_picture = !ws->picture_in_picture;
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
            const std::string version = "Version " + std::string(mj_versionString());
            ImGui::MenuItem(version.c_str());
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
        const studio::DialogResult res = studio::SaveFileDialog(ws->filename);
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
                const auto pixels = render_to_texture(ws, gfx, &ws->cam, width, height);
                save_webp(width, height, pixels.data(), res.path);
                break;
            }
            case WindowState::kPrintModel:
                mj_printModel(m, res.path.c_str());
                break;
            case WindowState::kPrintData:
                mj_printData(m, ws->data.get(), res.path.c_str());
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
    if (!ws->style_editor) studio::SetupTheme(ws->theme);
    ImGui::GetIO().FontGlobalScale = ws->font_scale;

    const ImVec4 workspace_rect = studio::ConfigureDockingLayout(ws->toolbar, ws->status_bar);
    mjModel     *m              = ws->model.get();
    mjData      *d              = ws->data.get();
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
            section("RecordingGui", "Recording", [&] { recording_gui(ws, cmd); });
        }
        ImGui::End();
    }

    if (ws->inspector_panel) {
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

    if (ws->picture_in_picture) {
        if (ImGui::Begin("Picture-in-Picture", &ws->picture_in_picture)) {
            const auto render_cb = [ws, &gfx](const studio::PipState &pip, int width, int height) {
                mjtByte  *flags        = ws->render_flags;
                const int prev_depth   = flags[mjRND_DEPTH];
                const int prev_segment = flags[mjRND_SEGMENT];
                flags[mjRND_DEPTH]     = pip.mode == studio::PipState::Depth ? 1 : 0;
                flags[mjRND_SEGMENT]   = pip.mode == studio::PipState::Segmentation ? 1 : 0;
                mjvCamera camera;
                mjv_defaultCamera(&camera);
                camera.type          = mjCAMERA_FIXED;
                camera.fixedcamid    = pip.camera;
                const auto output    = render_to_texture(ws, gfx, &camera, width, height);
                const auto id        = gfx.bridge->UploadImage(pip.texture, output, width, height);
                flags[mjRND_DEPTH]   = prev_depth;
                flags[mjRND_SEGMENT] = prev_segment;
                return static_cast<ImTextureID>(id);
            };
            studio::PipGui(m, gfx.window->GetAspectRatio(), render_cb, &ws->pips);
        }
        ImGui::End();
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

    file_dialog_gui(ws, cmd, gfx);

    if (ws->style_editor) {
        if (ImGui::Begin("Style Editor", &ws->style_editor)) ImGui::ShowStyleEditor();
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
        ws->scene->Sync(
          ws->model.get(), ws->data.get(), viewport, &ws->opt, &ws->pert, &ws->cam, extra
        );
    }
    gfx.bridge->Update();

    std::vector<mujoco::Renderer::View> views;
    if (ws->scene) views.push_back(scene_view(ws, gfx.renderer, &ws->cam));
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
    const studio::GraphicsMode mode = graphics_mode();
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
    studio::Window window(impl->title, kWindowWidth, kWindowHeight, { .gfx_mode = mode });

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
    renderer.CreateRenderTarget("tmp", target);
    renderer.GetScene("scene");
    mujoco::ImguiBridge bridge(renderer.GetContext(), renderer.GetScene("ux"));
    ImPlot::CreateContext();
    const Gfx gfx{ &window, &renderer, &bridge };

    // Studio's own settings file, so the layout and theme carry over between the two.
    WindowState ws;
    ws.ini_path = studio::GetDefaultIniPath();
    ws.theme    = studio::LoadSettings(ws.ini_path, studio::GuiTheme::kDark);
    mjv_defaultOption(&ws.opt);
    mjv_defaultPerturb(&ws.pert);
    mjv_makeScene(nullptr, &ws.plugin_scene, 2000);
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

    // Filament objects belong to the thread that made them, so they go here, scene first.
    for (studio::PipState &pip : ws.pips)
        if (pip.texture != ImTextureID_Invalid) bridge.UploadImage(pip.texture, {}, 0, 0);
    ws.scene.reset();
    ws.data.reset();
    mjv_freeScene(&ws.plugin_scene);
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

    auto *impl   = new ViewerImpl();
    impl->owner  = v;
    impl->env    = env;
    impl->title  = title ? title : "MuJoCo";
    impl->camera = v->cam;
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
