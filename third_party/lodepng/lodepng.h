// Minimal lodepng stub — only the encode() overload used by simulate.cc.
// Screenshots are silently skipped; all other simulate functionality works.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

// simulate.cc uses the unqualified name LCT_RGB so it must be in global scope
enum LodePNGColorType { LCT_GREY=0, LCT_RGB=2, LCT_RGBA=6 };

namespace lodepng {

inline unsigned encode(const std::string& /*filename*/,
                       const uint8_t* /*image*/,
                       unsigned /*w*/, unsigned /*h*/,
                       LodePNGColorType /*colortype*/ = LCT_RGBA,
                       unsigned /*bitdepth*/ = 8)
{
    // no-op: screenshot saving not supported in this build
    return 0; // 0 = success so simulate doesn't call mju_error
}

} // namespace lodepng
