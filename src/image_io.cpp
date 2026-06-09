/* ffmpeg-backed image writer used by the Simulate UI screenshot path.
 * Requires ffmpeg in PATH at runtime. */

#include "mj_kdl_wrapper/image_io.hpp"

#include <cstdio>

namespace mj_kdl {

bool write_png_rgb(const std::string &path, const std::uint8_t *rgb, int width, int height)
{
    if (!rgb || width <= 0 || height <= 0) return false;

    /* Build the ffmpeg command.
     * -f rawvideo       : input is raw pixels
     * -pix_fmt rgb24    : input format
     * -s WxH            : frame dimensions
     * -i pipe:0         : read from stdin
     * -vframes 1        : encode one frame
     * -y                : overwrite output without asking */
    char cmd[1024];
    std::snprintf(
      cmd,
      sizeof(cmd),
      "ffmpeg -loglevel error -f rawvideo -pix_fmt rgb24 -s %dx%d -i pipe:0 -vframes 1 -y \"%s\"",
      width,
      height,
      path.c_str()
    );

    FILE *pipe = popen(cmd, "w");
    if (!pipe) return false;

    const std::size_t n       = static_cast<std::size_t>(width) * height * 3;
    const std::size_t written = std::fwrite(rgb, 1, n, pipe);
    const int         rc      = pclose(pipe);

    return written == n && rc == 0;
}

} // namespace mj_kdl
