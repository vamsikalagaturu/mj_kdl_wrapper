#pragma once

#include <cstdint>
#include <string>

namespace mj_kdl {

bool write_png_rgb(const std::string &path, const std::uint8_t *rgb, int width, int height);

} // namespace mj_kdl
