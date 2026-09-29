/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#include "mj_kdl_wrapper/image_io.hpp"

#include <cerrno>
#include <csignal>
#include <string>
#include <vector>

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

namespace mj_kdl {

bool write_png_rgb(const std::string &path, const std::uint8_t *rgb, int width, int height)
{
    if (!rgb || width <= 0 || height <= 0) return false;

    // An argument list, no shell: the path reaches ffmpeg as written.
    const std::string        size = std::to_string(width) + "x" + std::to_string(height);
    std::vector<std::string> args = { "ffmpeg",   "-loglevel", "error",     "-f", "rawvideo",
                                      "-pix_fmt", "rgb24",     "-s",        size, "-i",
                                      "pipe:0",   "-y",        "-frames:v", "1",  path };
    std::vector<char *>      argv;
    for (auto &arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);

    // A missing or failing ffmpeg must cost the screenshot, not the process.
    std::signal(SIGPIPE, SIG_IGN);

    int fds[2];
    if (pipe2(fds, O_CLOEXEC) != 0) return false;
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, fds[0], STDIN_FILENO);
    pid_t     pid = -1;
    const int err = posix_spawnp(&pid, "ffmpeg", &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(fds[0]);
    if (err != 0) {
        close(fds[1]);
        return false;
    }

    const auto *bytes = rgb;
    std::size_t left  = static_cast<std::size_t>(width) * height * 3;
    while (left > 0) {
        const ssize_t n = write(fds[1], bytes, left);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        bytes += n;
        left -= static_cast<std::size_t>(n);
    }
    close(fds[1]);
    int wstatus = 0;
    while (waitpid(pid, &wstatus, 0) < 0 && errno == EINTR) {}
    return left == 0 && WIFEXITED(wstatus) && WEXITSTATUS(wstatus) == 0;
}

} // namespace mj_kdl
