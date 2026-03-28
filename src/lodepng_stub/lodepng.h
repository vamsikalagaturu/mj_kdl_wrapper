/* lodepng.h -- minimal stub; only what simulate.cc requires.
 * The actual encode() is in src/lodepng_ffmpeg.cpp (ffmpeg pipe, no lodepng dep). */

#pragma once

#include <string>

typedef enum LodePNGColorType {
    LCT_GREY       = 0,
    LCT_RGB        = 2,
    LCT_PALETTE    = 3,
    LCT_GREY_ALPHA = 4,
    LCT_RGBA       = 6,
} LodePNGColorType;

namespace lodepng {

/* Write raw pixels to a PNG file.
 * Returns 0 on success, non-zero on failure (mirrors the lodepng convention). */
unsigned encode(
  const std::string   &filename,
  const unsigned char *in,
  unsigned             w,
  unsigned             h,
  LodePNGColorType     colortype = LCT_RGBA,
  unsigned             bitdepth  = 8
);

} // namespace lodepng
