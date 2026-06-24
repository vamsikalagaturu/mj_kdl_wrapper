#pragma once

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace mj_kdl_examples {
namespace fs = std::filesystem;

// Mirrored in python/mj_kdl_wrapper/menagerie.py:_cache_root and CMakeLists.txt's
// _MJ_KDL_CACHE_ROOT; keep all three in sync.
inline fs::path cache_root()
{
    if (const char *xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
        return fs::path(xdg) / "mj_kdl_wrapper";
    }
    if (const char *home = std::getenv("HOME"); home && *home) {
        return fs::path(home) / ".cache" / "mj_kdl_wrapper";
    }
    return {};
}

inline std::string find_menagerie_model(const fs::path &relative)
{
    if (const char *root = std::getenv("MJ_KDL_MENAGERIE"); root && *root) {
        const auto path = fs::path(root) / relative;
        if (fs::exists(path)) return path.string();
    }
    const auto path = cache_root() / "menagerie" / relative;
    return fs::exists(path) ? path.string() : "";
}

inline std::string menagerie_model(const fs::path &relative)
{
    if (std::string path = find_menagerie_model(relative); !path.empty()) return path;
    throw std::runtime_error(
        relative.string() + " not found. Searched $MJ_KDL_MENAGERIE and "
        + (cache_root() / "menagerie").string()
        + ". Run 'mj-kdl-fetch-menagerie' or set MJ_KDL_MENAGERIE.");
}

inline std::string find_asset(const fs::path &relative)
{
    const auto path = cache_root() / "assets" / relative;
    return fs::exists(path) ? path.string() : "";
}

inline std::string asset(const fs::path &relative)
{
    if (std::string path = find_asset(relative); !path.empty()) return path;
    throw std::runtime_error(
        relative.string() + " not found in " + (cache_root() / "assets").string()
        + ". Run 'mj-kdl-fetch-menagerie' to populate bundled assets.");
}
} // namespace mj_kdl_examples
