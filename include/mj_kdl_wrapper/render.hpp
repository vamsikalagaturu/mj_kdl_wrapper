/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#pragma once

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <cstdint>
#include <functional>

namespace mj_kdl {

/**
 * @ingroup grp_recorder
 * 16:9 output resolution presets for init_video_recorder().
 */
enum class VideoResolution {
    R360p  = 360,  // 640  x 360
    R480p  = 480,  // 854  x 480
    R720p  = 720,  // 1280 x 720
    R1080p = 1080, // 1920 x 1080
    R2K    = 1440, // 2560 x 1440
    R4K    = 2160, // 3840 x 2160
};

/**
 * @ingroup grp_recorder
 * Headless video recorder of the mj_kdl_wrapper::render component. Renders frames offscreen with
 * MuJoCo's Filament renderer and pipes raw RGB data to an ffmpeg process, producing an H.264
 * MP4 without a display server or a window.
 *
 * Each recorder renders on a thread of its own: record_frame() and render_rgb() take a
 * snapshot of the Env's state, its overlay geoms, cam and opt on the caller's thread and return,
 * so a control loop pays for the snapshot, not the render.
 *
 * Requirements: a GPU with EGL (or Filament's software rasterizer) and ffmpeg in PATH.
 *
 * Typical usage:
 *
 *   VideoRecorder vr;
 *   init_video_recorder(&vr, env.model, "sim.mp4", VideoResolution::R1080p);
 *   vr.cam.azimuth = 135;  vr.cam.elevation = -20;  vr.cam.distance = 2.5;
 *
 *   for (int i = 0; i < steps; ++i) {
 *       step(&env);
 *       record_frame(&vr, &env);
 *   }
 *
 *   cleanup(&vr);
 */
struct VideoRecorder
{
    mjvCamera cam{};           // camera configuration; modify freely between frames
    mjvOption opt{};           // rendering options; modify freely between frames
    void     *_impl = nullptr; // opaque renderer + ffmpeg state
};

/**
 * @ingroup grp_recorder
 * Initialise a headless video recorder: an offscreen Filament renderer and an ffmpeg process
 * (H.264/MP4) fed through a pipe. The renderer follows the Env it records, a rebuilt model
 * included.
 *
 * @param vr        VideoRecorder to initialise; freed by cleanup(VideoRecorder*).
 * @param model     MuJoCo model whose default free camera vr->cam starts from.
 * @param out_path  Output MP4 path (e.g. "sim.mp4").
 * @param width     Frame width in pixels (default 1280).
 * @param height    Frame height in pixels (default 720).
 * @param fps       Playback frame rate (default 60).
 * @return an error if the renderer or ffmpeg cannot be started.
 */
Status init_video_recorder(
  VideoRecorder *vr,
  mjModel       *model,
  const char    *out_path,
  int            width  = 1280,
  int            height = 720,
  int            fps    = 60
);

/**
 * @ingroup grp_recorder
 * Initialise a VideoRecorder from a resolution preset, at 16:9.
 *
 * @param vr         VideoRecorder to initialise.
 * @param model      MuJoCo model.
 * @param out_path   Output MP4 path.
 * @param resolution VideoResolution preset (e.g. VideoResolution::R1080p).
 * @param fps        Playback frame rate (default 60).
 * @return an empty Status on success, else the error.
 */
Status init_video_recorder(
  VideoRecorder  *vr,
  mjModel        *model,
  const char     *out_path,
  VideoResolution resolution,
  int             fps = 60
);

/**
 * @ingroup grp_recorder
 * Snapshot env's current state as the next video frame; it is rendered and encoded on the
 * recorder's thread. Waits only while the previous frame has not been picked up.
 *
 * @param vr   VideoRecorder initialised by init_video_recorder().
 * @param env  Env to render.
 * @return false once rendering or the pipe to ffmpeg has failed; every frame is recorded.
 */
bool record_frame(VideoRecorder *vr, Env *env);

/**
 * @ingroup grp_recorder
 * Initialise offscreen rendering only: no ffmpeg process and no output file.
 * Use with render_rgb() to grab frames; the camera is vr->cam, as for a recording, and
 * cleanup(VideoRecorder*) frees it. An open vr is freed first.
 *
 * @param vr      VideoRecorder to initialise; freed by cleanup(VideoRecorder*).
 * @param model   MuJoCo model whose default free camera vr->cam starts from.
 * @param width   Frame width in pixels.
 * @param height  Frame height in pixels.
 * @return an error if the renderer cannot be started.
 */
Status init_offscreen(VideoRecorder *vr, mjModel *model, int width, int height);

/**
 * @ingroup grp_recorder
 * Render env's current state into a caller-owned top-down RGB8 buffer
 * of width*height*3 bytes.
 *
 * @param vr   VideoRecorder initialised by init_offscreen() or init_video_recorder().
 * @param env  Env to render.
 * @param out  Destination buffer, width*height*3 bytes.
 * @return true on success.
 */
bool render_rgb(VideoRecorder *vr, Env *env, std::uint8_t *out);

/**
 * @ingroup grp_recorder
 * Snapshot env's current state and return; done gets the top-down RGB8 frame on the recorder's
 * thread, valid only during the call. Waits only while the previous frame has not been picked up.
 *
 * @param vr    VideoRecorder initialised by init_offscreen() or init_video_recorder().
 * @param env   Env to render.
 * @param done  Receives width*height*3 bytes.
 * @return false once rendering has failed.
 */
bool render_rgb(VideoRecorder *vr, Env *env, std::function<void(const std::uint8_t *rgb)> done);

/**
 * @ingroup grp_recorder
 * Render the frame still waiting, finalise the MP4 and release the renderer.
 * After this call vr->_impl is null and the VideoRecorder may be discarded.
 *
 * @param vr  VideoRecorder to tear down.
 */
void cleanup(VideoRecorder *vr);

} // namespace mj_kdl
