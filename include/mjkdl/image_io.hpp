#pragma once

#include <cstdint>
#include <string>

namespace mjkdl {

bool write_png_rgb(const std::string &path, const std::uint8_t *rgb, int width, int height);

} // namespace mjkdl
