#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

namespace mjkdl_examples {
namespace fs = std::filesystem;

inline std::string find_asset(const fs::path &relative)
{
    const auto path = fs::path(MJKDL_ASSETS_DIR) / relative;
    return fs::exists(path) ? path.string() : "";
}

inline std::string asset(const fs::path &relative)
{
    if (std::string path = find_asset(relative); !path.empty()) return path;
    throw std::runtime_error(relative.string() + " not found in " MJKDL_ASSETS_DIR);
}
} // namespace mjkdl_examples
