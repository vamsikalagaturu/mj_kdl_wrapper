/* lodepng_ffmpeg.cpp -- ffmpeg-backed PNG encoder that satisfies the lodepng::encode()
 * call in simulate.cc.  No lodepng library is required; only ffmpeg in PATH. */

#include "lodepng_stub/lodepng.h"

#include <cstdio>
#include <string>

namespace lodepng {

unsigned encode(
  const std::string   &filename,
  const unsigned char *in,
  unsigned             w,
  unsigned             h,
  LodePNGColorType     colortype,
  unsigned /* bitdepth */
)
{
    /* Only RGB and RGBA are supported (simulate.cc always passes LCT_RGB). */
    const char *pix_fmt         = nullptr;
    unsigned    bytes_per_pixel = 0;
    if (colortype == LCT_RGB) {
        pix_fmt         = "rgb24";
        bytes_per_pixel = 3;
    } else if (colortype == LCT_RGBA) {
        pix_fmt         = "rgba";
        bytes_per_pixel = 4;
    } else {
        return 1; /* unsupported */
    }

    /* Build the ffmpeg command.
     * -f rawvideo       : input is raw pixels
     * -pix_fmt <fmt>    : rgb24 or rgba
     * -s WxH            : frame dimensions
     * -i pipe:0         : read from stdin
     * -vframes 1        : encode one frame
     * -y                : overwrite output without asking */
    char cmd[1024];
    std::snprintf(
      cmd,
      sizeof(cmd),
      "ffmpeg -loglevel error -f rawvideo -pix_fmt %s -s %ux%u -i pipe:0 -vframes 1 -y \"%s\"",
      pix_fmt,
      w,
      h,
      filename.c_str()
    );

    FILE *pipe = popen(cmd, "w");
    if (!pipe) { return 1; }

    const std::size_t n       = static_cast<std::size_t>(w) * h * bytes_per_pixel;
    const std::size_t written = std::fwrite(in, 1, n, pipe);
    const int         rc      = pclose(pipe);

    if (written != n || rc != 0) { return 1; }
    return 0;
}

} // namespace lodepng
