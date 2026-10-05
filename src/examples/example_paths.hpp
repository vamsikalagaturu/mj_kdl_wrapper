#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

namespace mj_kdl_examples {
namespace fs = std::filesystem;

inline std::string find_asset(const fs::path &relative)
{
    const auto path = fs::path(MJ_KDL_ASSETS_DIR) / relative;
    return fs::exists(path) ? path.string() : "";
}

inline std::string asset(const fs::path &relative)
{
    if (std::string path = find_asset(relative); !path.empty()) return path;
    throw std::runtime_error(relative.string() + " not found in " MJ_KDL_ASSETS_DIR);
}
} // namespace mj_kdl_examples
