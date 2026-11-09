/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#include "mj_kdl_wrapper/render.hpp"


#include <mujoco/mjrfilament.h>
#include "render/filament/support/model_decorations.h"
#include "render/filament/support/model_lights.h"
#include "render/filament/support/model_renderables.h"
#include "render/filament/support/model_resources.h"
#include "render/filament/support/renderer.h"

#include <algorithm>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <dlfcn.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

namespace mj_kdl {

// Pixel layout of every frame: RGB8, as ffmpeg's rgb24 input takes it.
static constexpr int kRgbBytesPerPixel = 3;

// Asset providers

static std::filesystem::path asset_dir()
{
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&asset_dir), &info) && info.dli_fname)
        return std::filesystem::path(info.dli_fname).parent_path() / "mujoco_assets";
    return "mujoco_assets";
}

struct AssetFile
{
    std::vector<char> bytes;
};

static int open_asset(mjResource *resource)
{
    const std::string_view name    = resource->name;
    const std::string_view subpath = name.substr(name.find(':') + 1);
    const auto             path    = asset_dir() / subpath;
    std::ifstream          in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        MJ_LOG_ERROR("render: cannot open asset " << path);
        return 0;
    }
    auto *file = new AssetFile();
    file->bytes.resize(static_cast<std::size_t>(in.tellg()));
    in.seekg(0);
    if (file->bytes.empty() || !in.read(file->bytes.data(), file->bytes.size())) {
        delete file;
        return 0;
    }
    resource->data = file;
    return static_cast<int>(file->bytes.size());
}

static int read_asset(mjResource *resource, const void **buffer)
{
    auto *file = static_cast<AssetFile *>(resource->data);
    *buffer    = file->bytes.data();
    return static_cast<int>(file->bytes.size());
}

static void close_asset(mjResource *resource)
{
    delete static_cast<AssetFile *>(resource->data);
    resource->data = nullptr;
}

/* Serve MuJoCo's "filament:" materials and "font:" files from the mujoco_assets directory next
 * to the render library, as mujoco_filament asks its user to (src/render/filament/CMakeLists.txt).
 * A prefix someone registered first (the mujoco Python package) is left to them: the engine's
 * provider table is per process and rejects a prefix registered twice. Also called by the viewer,
 * which declares it itself; not part of render.hpp. */
void register_asset_providers()
{
    static std::once_flag once;
    std::call_once(once, []() {
        mjpResourceProvider provider;
        mjp_defaultProvider(&provider);
        provider.open  = open_asset;
        provider.read  = read_asset;
        provider.close = close_asset;
        for (const char *prefix : { "filament", "font" }) {
            if (mjp_findProvider((std::string(prefix) + ":").c_str())) continue;
            provider.prefix = prefix;
            mjp_registerProvider(&provider);
        }
    });
}

// ffmpeg

// An ffmpeg process taking raw top-down RGB24 on stdin and writing H.264 MP4.
struct FfmpegSink
{
    FILE       *pipe = nullptr;
    pid_t       pid  = -1;
    std::string path;
    int         frames = 0;
};

static Status sink_open(FfmpegSink *sink, const char *out_path, int width, int height, int fps)
{
    // -g <fps>: ~1s GOP (vs x264's 250-frame default) so scrubbing seeks land near a keyframe.
    const int gop = std::max(1, fps);

    /* libx264 in yuv420p refuses an odd width or height, and a 16:9 width for an odd height can
       be odd, so drop the stray row or column here. */
    const int target_w = (width / 2) * 2;
    const int target_h = (height / 2) * 2;
    if (target_w <= 0 || target_h <= 0)
        MJ_FAIL("recording: frame is " << width << "x" << height << ", too small to encode");

    // An argument list, no shell: the path reaches ffmpeg as written, quotes and $(...) included.
    std::vector<std::string> args = {
        "ffmpeg",    "-hide_banner",
        "-loglevel", "error",
        "-nostats",  "-y",
        "-f",        "rawvideo",
        "-vcodec",   "rawvideo",
        "-pix_fmt",  "rgb24",
        "-s",        std::to_string(width) + "x" + std::to_string(height),
        "-r",        std::to_string(fps),
        "-i",        "pipe:0",
        "-an"
    };
    if (target_w != width || target_h != height) {
        args.insert(
          args.end(),
          { "-vf", "scale=" + std::to_string(target_w) + ":" + std::to_string(target_h) }
        );
    }
    const std::vector<std::string> encode = {
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",           "-preset", "medium",
        "-crf",    "18",      "-g",       std::to_string(gop), out_path
    };
    args.insert(args.end(), encode.begin(), encode.end());
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (auto &arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);

    /* An ffmpeg that dies mid-recording must not take the simulation down with it: without this
       the next write raises SIGPIPE, whose default is to kill the process. */
    std::signal(SIGPIPE, SIG_IGN);

    int fds[2];
    if (pipe2(fds, O_CLOEXEC) != 0) MJ_FAIL("pipe for ffmpeg failed: " << std::strerror(errno));
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, fds[0], STDIN_FILENO);
    const int err = posix_spawnp(&sink->pid, "ffmpeg", &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(fds[0]);
    if (err != 0) {
        close(fds[1]);
        MJ_FAIL("starting ffmpeg failed (" << std::strerror(err) << ") - is it in PATH?");
    }

    sink->pipe   = fdopen(fds[1], "w");
    sink->path   = out_path;
    sink->frames = 0;
    if (!sink->pipe) MJ_FAIL("opening the ffmpeg pipe failed: " << std::strerror(errno));
    return {};
}

static bool sink_write(FfmpegSink *sink, const void *rgb, std::size_t bytes)
{
    if (!sink->pipe) return false;
    if (fwrite(rgb, 1, bytes, sink->pipe) != bytes) {
        MJ_LOG_ERROR("fwrite to ffmpeg pipe failed");
        return false;
    }
    ++sink->frames;
    return true;
}

static void sink_close(FfmpegSink *sink)
{
    if (!sink->pipe) return;
    std::fclose(sink->pipe);
    sink->pipe  = nullptr;
    int wstatus = 0;
    waitpid(sink->pid, &wstatus, 0);
    sink->pid        = -1;
    const int status = WIFEXITED(wstatus) ? WEXITSTATUS(wstatus) : -1;
    if (status == 0 && sink->frames > 0) {
        MJ_LOG_INFO("recording saved to " << sink->path);
    } else if (status != 0) {
        MJ_LOG_ERROR("ffmpeg failed while saving recording to '" << sink->path << "'");
    }
}

// Recorder

using RgbDone = std::function<void(const std::uint8_t *rgb)>;

// One frame handed to the render thread: everything it draws, copied on the caller's thread.
struct RecorderJob
{
    enum class Kind { Record, Rgb, Callback } kind = Kind::Record;
    std::uint8_t                  *out             = nullptr; // Rgb: the caller's buffer
    RgbDone                        done;                      // Callback
    std::shared_ptr<const mjModel> model;                     // the recorder's copy
    std::uint64_t                  model_gen = 0;
    std::vector<mjtNum>            state; // mjSTATE_INTEGRATION
    mjvCamera                      cam{};
    mjvOption                      opt{};
    std::vector<mjvGeom>           overlays;
    std::uint64_t                  seq = 0;
};

// The retained Filament objects for one model, and the data the render thread poses them from.
struct ModelView
{
    std::shared_ptr<const mjModel>            model;
    std::uint64_t                             gen  = 0;
    mjData                                   *data = nullptr;
    std::unique_ptr<mujoco::ModelResources>   resources;
    std::unique_ptr<mujoco::ModelLights>      lights;
    std::unique_ptr<mujoco::ModelRenderables> renderables;
    std::unique_ptr<mujoco::ModelDecorations> decorations;
};

struct VideoRecorderImpl
{
    int        width  = 0;
    int        height = 0;
    FfmpegSink sink;

    // Render thread only: mjrf is not thread-safe, so every call stays on that thread.
    std::unique_ptr<mujoco::Renderer>    renderer;
    mujoco::Renderer::SceneHandle        scene;
    mujoco::Renderer::RenderTargetHandle target;
    ModelView                            view;

    // Caller's thread only.
    Env                           *env = nullptr; // the Env whose hooks keep model current
    std::shared_ptr<const mjModel> model;
    std::uint64_t                  model_gen   = 0;
    bool                           model_stale = true;

    std::thread             thread;
    std::mutex              mtx;
    std::condition_variable cv;
    RecorderJob             job;
    bool                    pending   = false;
    bool                    quit      = false;
    bool                    failed    = false; // sticky: rendering or the ffmpeg pipe gave out
    std::uint64_t           submitted = 0;
    std::uint64_t           finished  = 0;
    bool                    last_ok   = true; // result of job `finished`
};

static void release_view(ModelView *view)
{
    // Renderables and lights leave the scene as they go, before the resources they draw with.
    view->decorations.reset();
    view->renderables.reset();
    view->lights.reset();
    view->resources.reset();
    if (view->data) mj_deleteData(view->data);
    view->data  = nullptr;
    view->model = nullptr;
}

static bool make_view(VideoRecorderImpl *impl, const RecorderJob &job)
{
    ModelView &view = impl->view;
    release_view(&view);
    const mjModel *m     = job.model.get();
    mjrfContext   *ctx   = impl->renderer->GetContext();
    mjrfScene     *scene = impl->renderer->GetScene(impl->scene);
    view.data            = mj_makeData(m);
    if (!view.data) return false;
    view.model       = job.model;
    view.gen         = job.model_gen;
    view.resources   = std::make_unique<mujoco::ModelResources>(m, ctx);
    view.lights      = std::make_unique<mujoco::ModelLights>(scene, view.resources.get());
    view.renderables = std::make_unique<mujoco::ModelRenderables>(scene, view.resources.get());
    view.decorations = std::make_unique<mujoco::ModelDecorations>(ctx, scene, m);
    mjrf_configureSceneFromModel(scene, m);
    return true;
}

static bool render_job(VideoRecorderImpl *impl, const RecorderJob &job)
{
    if (!impl->view.model || impl->view.gen != job.model_gen) {
        if (!make_view(impl, job)) return false;
    }
    const mjModel *m = impl->view.model.get();
    mjData        *d = impl->view.data;
    mj_setState(m, d, job.state.data(), mjSTATE_INTEGRATION);
    mj_forward(m, d);

    mjvCamera     cam      = job.cam;
    const mjrRect viewport = { 0, 0, impl->width, impl->height };
    impl->view.lights->Update(d);
    impl->view.renderables->SetOptions(job.opt);
    impl->view.renderables->Update(d);
    impl->view.decorations->Update(d, &job.opt, nullptr, &cam, viewport, nullptr, job.overlays);

    mujoco::Renderer::View view;
    view.scene         = impl->scene;
    view.render_target = impl->target;
    view.camera        = mjv_camera2GLCamera(m, d, &cam);
    if (m->vis.headlight.active) {
        view.headlight = mujoco::Renderer::View::Headlight();
        std::copy_n(m->vis.headlight.diffuse, 3, view.headlight->color);
        view.headlight->intensity = impl->view.lights->GetHeadlightIntensity();
    }
    impl->renderer->Render({ &view, 1 });

    const std::span<const std::byte> px = impl->renderer->GetPixels(impl->target);
    switch (job.kind) {
    case RecorderJob::Kind::Record:
        return sink_write(&impl->sink, px.data(), px.size());
    case RecorderJob::Kind::Rgb:
        std::memcpy(job.out, px.data(), px.size());
        return true;
    case RecorderJob::Kind::Callback:
        job.done(reinterpret_cast<const std::uint8_t *>(px.data()));
        return true;
    }
    return false;
}

static void render_loop(VideoRecorderImpl *impl)
{
    register_asset_providers();
    mjrfContextConfig config;
    mjrf_defaultContextConfig(&config);
    config.graphics_api = mjGRAPHICS_API_OPENGL;
    impl->renderer      = std::make_unique<mujoco::Renderer>(config);
    impl->renderer->GetScene("scene");
    impl->scene = impl->renderer->GetSceneHandle("scene");

    mjrfRenderTargetConfig target;
    mjrf_defaultRenderTargetConfig(&target);
    target.width        = impl->width;
    target.height       = impl->height;
    target.color_format = mjPIXEL_FORMAT_RGB8;
    target.depth_format = mjPIXEL_FORMAT_DEPTH32F;
    impl->target        = impl->renderer->CreateRenderTarget("frame", target);

    for (;;) {
        std::unique_lock<std::mutex> lock(impl->mtx);
        impl->cv.wait(lock, [impl] { return impl->pending || impl->quit; });
        if (!impl->pending) break;
        RecorderJob job = std::move(impl->job);
        impl->pending   = false;
        lock.unlock();
        impl->cv.notify_all();

        const bool rendered = render_job(impl, job);
        lock.lock();
        impl->failed   = impl->failed || !rendered;
        impl->last_ok  = rendered;
        impl->finished = job.seq;
        lock.unlock();
        impl->cv.notify_all();
    }
    release_view(&impl->view);
    impl->renderer.reset();
}

Status init_offscreen(VideoRecorder *vr, mjModel *model, int width, int height)
{
    if (!vr || !model) MJ_FAIL("init_offscreen: null recorder or model");
    if (width <= 0 || height <= 0) MJ_FAIL("init_offscreen: " << width << "x" << height);
    cleanup(vr);

    mjv_defaultCamera(&vr->cam);
    mjv_defaultFreeCamera(model, &vr->cam);
    mjv_defaultOption(&vr->opt);

    auto *impl   = new VideoRecorderImpl();
    impl->width  = width;
    impl->height = height;
    impl->thread = std::thread(render_loop, impl);
    vr->_impl    = impl;
    return {};
}

static void follow_env(VideoRecorderImpl *impl, Env *env)
{
    if (impl->env == env) return;
    if (impl->env) impl->env->hooks.erase(impl);
    impl->env         = env;
    impl->model_stale = true;
    EnvHooks &hooks   = env->hooks[impl];
    hooks.on_rebuild  = [impl](Env *) { impl->model_stale = true; };
    hooks.on_close    = [impl](Env *) {
        impl->env         = nullptr;
        impl->model_stale = true;
    };
}

// Waits only until the render thread has picked up the previous frame.
static bool submit(VideoRecorder *vr, Env *env, RecorderJob job)
{
    auto                        *impl = static_cast<VideoRecorderImpl *>(vr->_impl);
    std::unique_lock<std::mutex> lock(impl->mtx);
    impl->cv.wait(lock, [impl] { return !impl->pending; });
    if (impl->failed) return false;
    lock.unlock();

    follow_env(impl, env);
    // The render thread draws from this copy, never from the Env's live model.
    if (impl->model_stale || !impl->model) {
        mjModel *copy = mj_copyModel(nullptr, env->model);
        if (!copy) {
            MJ_LOG_ERROR("recorder: copying the model failed");
            return false;
        }
        impl->model       = std::shared_ptr<const mjModel>(copy, [](const mjModel *m) {
            mj_deleteModel(const_cast<mjModel *>(m));
        });
        impl->model_stale = false;
        ++impl->model_gen;
    }
    job.state.resize(mj_stateSize(env->model, mjSTATE_INTEGRATION));
    mj_getState(env->model, env->data, job.state.data(), mjSTATE_INTEGRATION);
    job.cam       = vr->cam;
    job.opt       = vr->opt;
    job.overlays  = overlay_geoms(env);
    job.model     = impl->model;
    job.model_gen = impl->model_gen;

    lock.lock();
    job.seq       = ++impl->submitted;
    impl->job     = std::move(job);
    impl->pending = true;
    lock.unlock();
    impl->cv.notify_all();
    return true;
}

Status init_video_recorder(
  VideoRecorder *vr,
  mjModel       *model,
  const char    *out_path,
  int            width,
  int            height,
  int            fps
)
{
    if (!out_path) MJ_FAIL("init_video_recorder: null output path");
    if (Status s = init_offscreen(vr, model, width, height); !s) return s;

    auto *impl = static_cast<VideoRecorderImpl *>(vr->_impl);
    if (Status s = sink_open(&impl->sink, out_path, width, height, fps); !s) {
        cleanup(vr);
        return s;
    }

    MJ_LOG_INFO(
      "VideoRecorder: " << width << "x" << height << " @ " << fps << " fps -> " << out_path
    );
    return {};
}

Status init_video_recorder(
  VideoRecorder  *vr,
  mjModel        *model,
  const char     *out_path,
  VideoResolution resolution,
  int             fps
)
{
    // 16:9 at the named height, rounded to even (480p is 853.3 wide; the encoder takes no odd).
    const int h = static_cast<int>(resolution);
    const int w = ((h * 16 / 9) + 1) / 2 * 2;
    return init_video_recorder(vr, model, out_path, w, h, fps);
}

bool render_rgb(VideoRecorder *vr, Env *env, std::uint8_t *out)
{
    if (!vr || !vr->_impl || !env || !env->model || !out) return false;
    auto       *impl = static_cast<VideoRecorderImpl *>(vr->_impl);
    RecorderJob job;
    job.kind = RecorderJob::Kind::Rgb;
    job.out  = out;
    if (!submit(vr, env, std::move(job))) return false;
    std::unique_lock<std::mutex> lock(impl->mtx);
    const std::uint64_t          seq = impl->submitted;
    impl->cv.wait(lock, [impl, seq] { return impl->finished >= seq; });
    return impl->last_ok;
}

bool render_rgb(VideoRecorder *vr, Env *env, std::function<void(const std::uint8_t *rgb)> done)
{
    if (!vr || !vr->_impl || !env || !env->model || !done) return false;
    RecorderJob job;
    job.kind = RecorderJob::Kind::Callback;
    job.done = std::move(done);
    return submit(vr, env, std::move(job));
}

bool record_frame(VideoRecorder *vr, Env *env)
{
    if (!vr || !vr->_impl || !env || !env->model) return false;
    if (!static_cast<VideoRecorderImpl *>(vr->_impl)->sink.pipe) return false;
    return submit(vr, env, RecorderJob{});
}

void cleanup(VideoRecorder *vr)
{
    if (!vr || !vr->_impl) return;
    auto *impl = static_cast<VideoRecorderImpl *>(vr->_impl);
    {
        // The frame still waiting is rendered before the thread ends.
        const std::lock_guard<std::mutex> lock(impl->mtx);
        impl->quit = true;
    }
    impl->cv.notify_all();
    if (impl->thread.joinable()) impl->thread.join();
    if (impl->env) impl->env->hooks.erase(impl);
    sink_close(&impl->sink);
    delete impl;
    vr->_impl = nullptr;
}

} // namespace mj_kdl
