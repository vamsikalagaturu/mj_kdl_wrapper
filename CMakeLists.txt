cmake_minimum_required(VERSION 3.16)

include(cmake/Versions.cmake)

project(mj_kdl_wrapper VERSION ${MJ_KDL_VERSION} LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# The installed bundled models are cached here.
if(DEFINED ENV{XDG_CACHE_HOME} AND NOT "$ENV{XDG_CACHE_HOME}" STREQUAL "")
    set(_MJ_KDL_CACHE_ROOT "$ENV{XDG_CACHE_HOME}/mj_kdl_wrapper")
elseif(DEFINED ENV{HOME} AND NOT "$ENV{HOME}" STREQUAL "")
    set(_MJ_KDL_CACHE_ROOT "$ENV{HOME}/.cache/mj_kdl_wrapper")
endif()

# MuJoCo is built from the source checkout `vcs import < mj_kdl_wrapper.repos` places here: the
# release archive ships neither the Filament renderer nor the Studio libraries as libraries.
set(
  MJ_KDL_MUJOCO_DIR "${CMAKE_SOURCE_DIR}/third_party/mujoco"
  CACHE PATH
  "MuJoCo source checkout"
)

# Orocos KDL is always the secorolab fork built from source - it is the only KDL this
# project uses; no system liborocos-kdl is ever consulted. `vcs import <
# mj_kdl_wrapper.repos` checks it out at the default location.
set(
  MJ_KDL_OROCOS_KDL_DIR "${CMAKE_SOURCE_DIR}/third_party/orocos_kinematics_dynamics"
  CACHE PATH
  "Orocos KDL fork source checkout"
)
# Consume an already-installed Orocos KDL prefix instead of building the fork.
# Intended for ROS 2 / colcon, where the secorolab fork is a workspace package
# and a single shared liborocos-kdl (and PyKDL) must be used everywhere to avoid
# loading two copies with the same SONAME in one process. When set, no
# ExternalProject build runs and nothing is bundled; the prefix must contain the
# secorolab fork. Takes precedence over MJ_KDL_OROCOS_KDL_DIR.
set(
  MJ_KDL_OROCOS_KDL_INSTALL_DIR ""
  CACHE PATH
  "Pre-installed Orocos KDL prefix to consume (skips building/bundling the fork)"
)
# Consume an Orocos KDL provided as a CMake package on CMAKE_PREFIX_PATH via
# find_package(orocos_kdl). This is the natural way to share one KDL across a
# workspace - a colcon overlay (the fork built as the orocos_kdl package) or any
# prefix on CMAKE_PREFIX_PATH where the secorolab fork is installed. When ON, no
# ExternalProject build runs and nothing is bundled. The found package MUST be the
# secorolab fork, not a stock distro KDL. Takes precedence over
# MJ_KDL_OROCOS_KDL_INSTALL_DIR and the in-tree build.
option(
  MJ_KDL_OROCOS_KDL_FROM_PACKAGE
  "Consume Orocos KDL via find_package(orocos_kdl) instead of building the fork"
  OFF
)

# Optional components on top of the core library, which renders nothing and opens no window.
# render: offscreen Filament rendering (VideoRecorder); viewer: the Studio window, which records
# through render. AUTO builds one when the system libraries it needs are found.
set(MJ_KDL_WITH_RENDER "AUTO" CACHE STRING "Build the render target: AUTO, ON, or OFF")
set_property(CACHE MJ_KDL_WITH_RENDER PROPERTY STRINGS AUTO ON OFF)
set(MJ_KDL_WITH_VIEWER "AUTO" CACHE STRING "Build the viewer target: AUTO, ON, or OFF")
set_property(CACHE MJ_KDL_WITH_VIEWER PROPERTY STRINGS AUTO ON OFF)

set(MJ_KDL_RENDER_ENABLED OFF)
if(NOT MJ_KDL_WITH_RENDER STREQUAL "OFF")
    # Filament renders headless through EGL.
    if(MJ_KDL_WITH_RENDER STREQUAL "AUTO")
        find_package(OpenGL QUIET COMPONENTS EGL)
    else()
        find_package(OpenGL REQUIRED COMPONENTS EGL)
    endif()
    if(OpenGL_EGL_FOUND)
        set(MJ_KDL_RENDER_ENABLED ON)
    endif()
endif()

set(MJ_KDL_VIEWER_ENABLED OFF)
if(NOT MJ_KDL_WITH_VIEWER STREQUAL "OFF")
    if(MJ_KDL_WITH_VIEWER STREQUAL "ON" AND NOT MJ_KDL_RENDER_ENABLED)
        message(FATAL_ERROR "MJ_KDL_WITH_VIEWER=ON needs the render target (MJ_KDL_WITH_RENDER)")
    endif()
    # SDL2, which MuJoCo builds statically, opens its window through X11.
    if(MJ_KDL_WITH_VIEWER STREQUAL "AUTO")
        find_package(X11 QUIET)
    else()
        find_package(X11 REQUIRED)
    endif()
    if(X11_FOUND AND MJ_KDL_RENDER_ENABLED)
        set(MJ_KDL_VIEWER_ENABLED ON)
    endif()
endif()
message(STATUS "render : ${MJ_KDL_RENDER_ENABLED} (MJ_KDL_WITH_RENDER=${MJ_KDL_WITH_RENDER})")
message(STATUS "viewer : ${MJ_KDL_VIEWER_ENABLED} (MJ_KDL_WITH_VIEWER=${MJ_KDL_WITH_VIEWER})")

if(DEFINED SKBUILD)
# A wheel links the libmujoco and headers of the pip mujoco-core package (a build requirement),
# the same library it loads at runtime through $ORIGIN/../mujoco.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))"
  OUTPUT_VARIABLE MUJOCO_LIB_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
set(_MJ_KDL_MUJOCO_LIB "${MUJOCO_LIB_DIR}/libmujoco.so.${MJ_KDL_MUJOCO_VERSION}")
if(NOT EXISTS "${_MJ_KDL_MUJOCO_LIB}")
    message(FATAL_ERROR "The mujoco package at ${MUJOCO_LIB_DIR} is not MuJoCo "
      "${MJ_KDL_MUJOCO_VERSION} (no libmujoco.so.${MJ_KDL_MUJOCO_VERSION}).")
endif()
add_library(mujoco::mujoco SHARED IMPORTED GLOBAL)
set_target_properties(mujoco::mujoco PROPERTIES
  IMPORTED_LOCATION "${_MJ_KDL_MUJOCO_LIB}"
  IMPORTED_SONAME "libmujoco.so.${MJ_KDL_MUJOCO_VERSION}"
  INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_LIB_DIR}/include"
)
# Importing mujoco registers its plugins, so the wrapper never loads any itself.
set(MJ_KDL_MUJOCO_PLUGIN_DIR "")
message(STATUS "MuJoCo : pip package ${MUJOCO_LIB_DIR}")
endif()

# The source build supplies libmujoco to a C++ build, and the Filament and Studio archives to
# both; a wheel takes only the archives and links them against the pip package's libmujoco,
# the same 4.0.0 library, so a process never loads two.
if(NOT DEFINED SKBUILD OR MJ_KDL_RENDER_ENABLED)
if(NOT EXISTS "${MJ_KDL_MUJOCO_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "No MuJoCo source at ${MJ_KDL_MUJOCO_DIR}. Run 'vcs import < mj_kdl_wrapper.repos' "
      "or point MJ_KDL_MUJOCO_DIR at a MuJoCo ${MJ_KDL_MUJOCO_VERSION} checkout.")
endif()
file(STRINGS "${MJ_KDL_MUJOCO_DIR}/include/mujoco/mujoco.h" MUJOCO_VERSION_HEADER_LINE
     REGEX "^#define mjVERSION_HEADER ")
string(REGEX REPLACE ".*mjVERSION_HEADER +([0-9]+).*" "\\1" MUJOCO_VERSION_HEADER
       "${MUJOCO_VERSION_HEADER_LINE}")
if(NOT MUJOCO_VERSION_HEADER STREQUAL "${MJ_KDL_MUJOCO_VERSION_HEADER}")
    message(FATAL_ERROR
      "mj_kdl_wrapper requires MuJoCo ${MJ_KDL_MUJOCO_VERSION} "
      "(mjVERSION_HEADER ${MJ_KDL_MUJOCO_VERSION_HEADER}), "
      "but found ${MUJOCO_VERSION_HEADER} at ${MJ_KDL_MUJOCO_DIR}")
endif()

# MuJoCo's CMake expects to be the top-level project (its dependency includes resolve through
# CMAKE_SOURCE_DIR), so it is built as one, as MuJoCo's own Python build does: a host build whose
# static Filament and Studio archives are imported below by the list python/mujoco/CMakeLists.txt
# links them with.
include(ExternalProject)
set(_MJ_KDL_MUJOCO_BIN "${CMAKE_BINARY_DIR}/mujoco")
set(_MJ_KDL_MUJOCO_ARCHIVE_DIR "${_MJ_KDL_MUJOCO_BIN}/lib")

set(_mj_kdl_mujoco_targets)
set(_mj_kdl_mujoco_byproducts)
if(NOT DEFINED SKBUILD)
    set(MUJOCO_LIB_DIR "${_MJ_KDL_MUJOCO_ARCHIVE_DIR}")
    set(MJ_KDL_MUJOCO_PLUGIN_DIR "${MUJOCO_LIB_DIR}/mujoco_plugin")
    set(_MJ_KDL_MUJOCO_LIB "${MUJOCO_LIB_DIR}/libmujoco.so.${MJ_KDL_MUJOCO_VERSION}")
    list(APPEND _mj_kdl_mujoco_targets mujoco actuator sensor sdf_plugin)
    list(APPEND _mj_kdl_mujoco_byproducts "${_MJ_KDL_MUJOCO_LIB}")
    foreach(_plugin actuator sensor sdf_plugin)
        list(APPEND _mj_kdl_mujoco_byproducts "${MJ_KDL_MUJOCO_PLUGIN_DIR}/lib${_plugin}.so")
    endforeach()
endif()
# In link order: each archive comes before the ones it needs.
set(_mj_kdl_filament_archives
  mujoco_filament dear_imgui filament backend image ktxreader utils math bluegl bluevk filabridge
  filaflat filamat matlang shaders smol-v glslang SPIRV SPIRV-Tools SPIRV-Tools-opt basis_encoder
  basis_transcoder zstd z
)
set(_mj_kdl_platform_archives
  mujoco_studio_platform dear_imgui_SDL2 dear_imgui_OpenGL3 dear_imgui_stdlib implot SDL2
  webp sharpyuv
)
if(MJ_KDL_RENDER_ENABLED)
    list(APPEND _mj_kdl_mujoco_targets mujoco_filament mujoco_filament_assets)
    foreach(_archive IN LISTS _mj_kdl_filament_archives)
        list(APPEND _mj_kdl_mujoco_byproducts "${_MJ_KDL_MUJOCO_ARCHIVE_DIR}/lib${_archive}.a")
    endforeach()
endif()
if(MJ_KDL_VIEWER_ENABLED)
    # webp writes Save Screenshot's file, as in Studio's own app.
    list(APPEND _mj_kdl_mujoco_targets mujoco_studio_platform mujoco_studio_assets webp)
    foreach(_archive IN LISTS _mj_kdl_platform_archives)
        list(APPEND _mj_kdl_mujoco_byproducts "${_MJ_KDL_MUJOCO_ARCHIVE_DIR}/lib${_archive}.a")
    endforeach()
endif()
set(_mj_kdl_mujoco_build_command ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release)
foreach(_target IN LISTS _mj_kdl_mujoco_targets)
    list(APPEND _mj_kdl_mujoco_build_command --target ${_target})
endforeach()

ExternalProject_Add(mujoco_ep
  DOWNLOAD_COMMAND ""
  SOURCE_DIR       "${MJ_KDL_MUJOCO_DIR}"
  BINARY_DIR       "${_MJ_KDL_MUJOCO_BIN}"
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DUSE_STATIC_LIBCXX=OFF
    -DMUJOCO_BUILD_EXAMPLES=OFF
    -DMUJOCO_BUILD_SIMULATE=OFF
    -DMUJOCO_BUILD_TESTS=OFF
    -DMUJOCO_TEST_PYTHON_UTIL=OFF
    -DMUJOCO_USE_FILAMENT=${MJ_KDL_RENDER_ENABLED}
    -DMUJOCO_BUILD_STUDIO=${MJ_KDL_VIEWER_ENABLED}
  BUILD_COMMAND    ${_mj_kdl_mujoco_build_command}
  INSTALL_COMMAND  ""
  # The host build's own dependency tracking decides what to rebuild.
  BUILD_ALWAYS     TRUE
  BUILD_BYPRODUCTS ${_mj_kdl_mujoco_byproducts}
)

set(_MJ_KDL_MUJOCO_DEPS "${_MJ_KDL_MUJOCO_BIN}/_deps")
if(NOT DEFINED SKBUILD)
    add_library(mujoco::mujoco SHARED IMPORTED GLOBAL)
    set_target_properties(mujoco::mujoco PROPERTIES
      IMPORTED_LOCATION "${_MJ_KDL_MUJOCO_LIB}"
      IMPORTED_SONAME "libmujoco.so.${MJ_KDL_MUJOCO_VERSION}"
      INTERFACE_INCLUDE_DIRECTORIES "${MJ_KDL_MUJOCO_DIR}/include"
    )
    add_dependencies(mujoco::mujoco mujoco_ep)
endif()

if(MJ_KDL_RENDER_ENABLED)
    set(_mj_kdl_filament_includes
      "${MJ_KDL_MUJOCO_DIR}/include"
      "${MJ_KDL_MUJOCO_DIR}/src"
      "${_MJ_KDL_MUJOCO_DEPS}/filament-src/libs/math/include"
    )
    # The host build fetches these; made up front so the targets' paths exist at configure time.
    file(MAKE_DIRECTORY ${_mj_kdl_filament_includes})
    set(_mj_kdl_filament_links)
    foreach(_archive IN LISTS _mj_kdl_filament_archives)
        add_library(mujoco::dep::${_archive} STATIC IMPORTED GLOBAL)
        set_target_properties(mujoco::dep::${_archive} PROPERTIES
          IMPORTED_LOCATION "${_MJ_KDL_MUJOCO_ARCHIVE_DIR}/lib${_archive}.a")
        add_dependencies(mujoco::dep::${_archive} mujoco_ep)
        list(APPEND _mj_kdl_filament_links mujoco::dep::${_archive})
    endforeach()
    add_library(mujoco::filament INTERFACE IMPORTED GLOBAL)
    set_target_properties(mujoco::filament PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${_mj_kdl_filament_includes}"
      INTERFACE_LINK_LIBRARIES "${_mj_kdl_filament_links};mujoco::mujoco;pthread;dl;rt"
    )
endif()

if(MJ_KDL_VIEWER_ENABLED)
    set(_mj_kdl_platform_includes
      "${MJ_KDL_MUJOCO_DIR}"
      "${_MJ_KDL_MUJOCO_DEPS}/dear_imgui-src"
      "${_MJ_KDL_MUJOCO_DEPS}/implot-src"
      "${_MJ_KDL_MUJOCO_DEPS}/sdl2-build/include/SDL2"
      "${_MJ_KDL_MUJOCO_DEPS}/sdl2-build/include-config-release/SDL2"
      "${_MJ_KDL_MUJOCO_DEPS}/libwebp-src/src"
    )
    file(MAKE_DIRECTORY ${_mj_kdl_platform_includes})
    set(_mj_kdl_platform_links)
    foreach(_archive IN LISTS _mj_kdl_platform_archives)
        add_library(mujoco::dep::${_archive} STATIC IMPORTED GLOBAL)
        set_target_properties(mujoco::dep::${_archive} PROPERTIES
          IMPORTED_LOCATION "${_MJ_KDL_MUJOCO_ARCHIVE_DIR}/lib${_archive}.a")
        add_dependencies(mujoco::dep::${_archive} mujoco_ep)
        list(APPEND _mj_kdl_platform_links mujoco::dep::${_archive})
    endforeach()
    add_library(mujoco::platform INTERFACE IMPORTED GLOBAL)
    set_target_properties(mujoco::platform PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${_mj_kdl_platform_includes}"
      INTERFACE_LINK_LIBRARIES "${_mj_kdl_platform_links};mujoco::filament"
    )
endif()
message(STATUS "MuJoCo : ${MJ_KDL_MUJOCO_DIR} -> ${_MJ_KDL_MUJOCO_BIN}")
endif()

# Build the pinned secorolab Orocos KDL fork at build time via ExternalProject
# (download + configure + build + install). This is the ONLY KDL the project uses;
# no system liborocos-kdl is ever consulted. The wrapper consumes it through an
# IMPORTED target and PyKDL (built later) consumes its install prefix.
include(ExternalProject)
# KDL's public headers include <Eigen/Core>, whichever KDL is used.
find_package(Eigen3 REQUIRED NO_MODULE)
# TRUE only when this build compiles and installs its own copy of the fork. The
# consume-from-elsewhere branches leave it FALSE, which switches off PyKDL
# building, KDL bundling, and the self-contained install of KDL.
set(_MJ_KDL_BUILD_OROCOS_KDL FALSE)
if(MJ_KDL_OROCOS_KDL_FROM_PACKAGE)
  # Consume an Orocos KDL provided as a CMake package on CMAKE_PREFIX_PATH (a
  # colcon overlay, or any prefix where the fork is installed). find_package
  # creates the orocos-kdl imported target and sets orocos_kdl_INCLUDE_DIRS.
  find_package(orocos_kdl REQUIRED)
  set(orocos_kdl_LIBRARIES orocos-kdl)
  set(orocos_kdl_DIR "found: orocos_kdl package")
  message(STATUS "Orocos KDL: using found orocos_kdl package")
elseif(MJ_KDL_OROCOS_KDL_INSTALL_DIR)
  # Consume a pre-installed Orocos KDL (e.g. a colcon workspace overlay). No
  # ExternalProject build, no bundling: a single shared liborocos-kdl is used by
  # the wrapper and by any other package in the same workspace/process.
  find_path(MJ_KDL_OROCOS_KDL_INCLUDE_DIR kdl/chain.hpp
            PATHS "${MJ_KDL_OROCOS_KDL_INSTALL_DIR}/include"
            NO_DEFAULT_PATH REQUIRED)
  find_library(MJ_KDL_OROCOS_KDL_LIBRARY orocos-kdl
               PATHS "${MJ_KDL_OROCOS_KDL_INSTALL_DIR}/lib"
               NO_DEFAULT_PATH REQUIRED)
  add_library(orocos-kdl SHARED IMPORTED GLOBAL)
  set_target_properties(orocos-kdl PROPERTIES
    IMPORTED_LOCATION "${MJ_KDL_OROCOS_KDL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MJ_KDL_OROCOS_KDL_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES Eigen3::Eigen
  )
  get_target_property(_mj_kdl_eigen_inc Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
  set(orocos_kdl_LIBRARIES    orocos-kdl)
  set(orocos_kdl_INCLUDE_DIRS "${MJ_KDL_OROCOS_KDL_INCLUDE_DIR}" ${_mj_kdl_eigen_inc})
  set(orocos_kdl_DIR          "installed: ${MJ_KDL_OROCOS_KDL_INSTALL_DIR}")
  message(STATUS "Orocos KDL: using installed prefix ${MJ_KDL_OROCOS_KDL_INSTALL_DIR}")
else()
set(_MJ_KDL_BUILD_OROCOS_KDL TRUE)
set(_MJ_KDL_OROCOS_PREFIX "${CMAKE_BINARY_DIR}/orocos_kdl_install")
set(_MJ_KDL_OROCOS_SRC "${MJ_KDL_OROCOS_KDL_DIR}")
if(NOT EXISTS "${_MJ_KDL_OROCOS_SRC}/orocos_kdl/CMakeLists.txt")
  message(FATAL_ERROR
    "No Orocos KDL source at ${_MJ_KDL_OROCOS_SRC}. Run "
    "'vcs import < mj_kdl_wrapper.repos', point MJ_KDL_OROCOS_KDL_DIR at a "
    "fork checkout, or set MJ_KDL_OROCOS_KDL_INSTALL_DIR. No system KDL is used.")
endif()
message(STATUS "Orocos KDL: building ${_MJ_KDL_OROCOS_SRC}")
ExternalProject_Add(orocos_kdl_ep
  DOWNLOAD_COMMAND ""
  SOURCE_DIR     "${_MJ_KDL_OROCOS_SRC}"
  SOURCE_SUBDIR  orocos_kdl
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${_MJ_KDL_OROCOS_PREFIX}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DENABLE_TESTS=OFF
    -DENABLE_EXAMPLES=OFF
    # Same flags as the wrapper: Eigen's allocation ABI in KDL's headers must match its library.
    "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
    "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
    "-DCMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}"
  BUILD_BYPRODUCTS "${_MJ_KDL_OROCOS_PREFIX}/lib/liborocos-kdl.so"
)
# IMPORTED target the wrapper links against. The install tree is produced by the
# ExternalProject build step; create the include dir up front so the path
# recorded on the target is valid at configure time.
file(MAKE_DIRECTORY "${_MJ_KDL_OROCOS_PREFIX}/include")
add_library(orocos-kdl SHARED IMPORTED GLOBAL)
set_target_properties(orocos-kdl PROPERTIES
  IMPORTED_LOCATION "${_MJ_KDL_OROCOS_PREFIX}/lib/liborocos-kdl.so"
  INTERFACE_INCLUDE_DIRECTORIES "${_MJ_KDL_OROCOS_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES Eigen3::Eigen
)
get_target_property(_mj_kdl_eigen_inc Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
set(orocos_kdl_LIBRARIES    orocos-kdl)
set(orocos_kdl_INCLUDE_DIRS "${_MJ_KDL_OROCOS_PREFIX}/include" ${_mj_kdl_eigen_inc})
set(orocos_kdl_DIR          "secorolab fork (ExternalProject)")
endif()

message(STATUS "KDL    : ${orocos_kdl_LIBRARIES}")
message(STATUS "KDL inc: ${orocos_kdl_INCLUDE_DIRS}")
message(STATUS "KDL cfg: ${orocos_kdl_DIR}")


# The default stage is compiled in, so a scene needs no file at run time (installed, wheel).
set(_mj_kdl_stage_xml "${CMAKE_CURRENT_SOURCE_DIR}/assets/stage/default.xml")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_mj_kdl_stage_xml}")
file(READ "${_mj_kdl_stage_xml}" _mj_kdl_stage)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/default_stage.cpp.tmp"
  "namespace mj_kdl {\nextern const char *const kDefaultStageXml;\n"
  "const char *const kDefaultStageXml = R\"mjkdl(${_mj_kdl_stage})mjkdl\";\n}\n")
configure_file("${CMAKE_CURRENT_BINARY_DIR}/generated/default_stage.cpp.tmp"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/default_stage.cpp" COPYONLY)

# Core library: MuJoCo and KDL only.
add_library(mj_kdl_wrapper SHARED
  src/mj_kdl_wrapper.cpp
  "${CMAKE_CURRENT_BINARY_DIR}/generated/default_stage.cpp"
)

target_include_directories(mj_kdl_wrapper PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# The installed package config recreates mujoco::mujoco, which cannot travel in the export.
target_link_libraries(mj_kdl_wrapper PUBLIC
  $<BUILD_INTERFACE:mujoco::mujoco>
  $<INSTALL_INTERFACE:mujoco::mujoco>
  ${orocos_kdl_LIBRARIES}
  # The installed fork's orocos-kdl target does not carry Eigen.
  Eigen3::Eigen
)
# DT_RPATH, not RUNPATH: LD_LIBRARY_PATH must not swap the KDL fork for a system liborocos-kdl
# with the same soname.
target_link_options(mj_kdl_wrapper PUBLIC "LINKER:--disable-new-dtags")
set_property(TARGET mj_kdl_wrapper APPEND PROPERTY BUILD_RPATH "${MUJOCO_LIB_DIR}")

# When KDL is built via ExternalProject, its headers and library only exist
# after that build step. Order it before the wrapper compiles/links.
if(TARGET orocos_kdl_ep)
    add_dependencies(mj_kdl_wrapper orocos_kdl_ep)
endif()
if(TARGET mujoco_ep)
    add_dependencies(mj_kdl_wrapper mujoco_ep)
endif()

target_compile_options(mj_kdl_wrapper PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(mj_kdl_wrapper PRIVATE
  MUJOCO_PLUGIN_DIR="${MJ_KDL_MUJOCO_PLUGIN_DIR}"
)
# dladdr() finds a mujoco_plugin directory installed next to the library.
target_link_libraries(mj_kdl_wrapper PRIVATE ${CMAKE_DL_LIBS})

# Filament's materials (and Studio's fonts, for the viewer) are read at runtime from
# mujoco_assets next to the render library; see register_asset_providers() in src/render.cpp.
set(MJ_KDL_RENDER_ASSETS_DIR "${CMAKE_CURRENT_BINARY_DIR}/mujoco_assets")

if(MJ_KDL_RENDER_ENABLED)
    add_library(mj_kdl_render SHARED src/render.cpp)
    add_library(mj_kdl_wrapper::render ALIAS mj_kdl_render)
    set_target_properties(mj_kdl_render PROPERTIES EXPORT_NAME render)
    target_link_libraries(mj_kdl_render PUBLIC mj_kdl_wrapper)
    target_link_libraries(mj_kdl_render PRIVATE mujoco::filament ${CMAKE_DL_LIBS})
    target_compile_options(mj_kdl_render PRIVATE -Wall -Wextra -Wpedantic)
    set_property(TARGET mj_kdl_render APPEND PROPERTY BUILD_RPATH "${MUJOCO_LIB_DIR}")

    add_custom_target(mj_kdl_render_assets
      COMMAND ${CMAKE_COMMAND} -E make_directory "${MJ_KDL_RENDER_ASSETS_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy_directory
              "${_MJ_KDL_MUJOCO_BIN}/src/render/filament/assets" "${MJ_KDL_RENDER_ASSETS_DIR}"
      DEPENDS mujoco_ep
      COMMENT "Staging Filament assets into ${MJ_KDL_RENDER_ASSETS_DIR}"
    )
    add_dependencies(mj_kdl_render mj_kdl_render_assets)
endif()

if(MJ_KDL_VIEWER_ENABLED)
    add_library(mj_kdl_viewer SHARED src/viewer.cpp)
    add_library(mj_kdl_wrapper::viewer ALIAS mj_kdl_viewer)
    set_target_properties(mj_kdl_viewer PROPERTIES EXPORT_NAME viewer)
    target_link_libraries(mj_kdl_viewer PUBLIC mj_kdl_wrapper mj_kdl_render)
    target_link_libraries(mj_kdl_viewer PRIVATE mujoco::platform)
    target_compile_options(mj_kdl_viewer PRIVATE -Wall -Wextra -Wpedantic)
    set_property(TARGET mj_kdl_viewer APPEND PROPERTY BUILD_RPATH "${MUJOCO_LIB_DIR}")

    add_custom_target(mj_kdl_viewer_assets
      COMMAND ${CMAKE_COMMAND} -E copy_directory
              "${_MJ_KDL_MUJOCO_BIN}/bin/assets" "${MJ_KDL_RENDER_ASSETS_DIR}"
      DEPENDS mujoco_ep mj_kdl_render_assets
      COMMENT "Staging Studio fonts into ${MJ_KDL_RENDER_ASSETS_DIR}"
    )
    add_dependencies(mj_kdl_viewer mj_kdl_viewer_assets)
endif()

# Optional ROS component: publishes an already-rendered camera frame as sensor_msgs/Image.
# AUTO builds it when rclcpp and sensor_msgs are both findable; a ROS-less configure simply
# leaves the target out, and the core library is unchanged either way.
set(MJ_KDL_WITH_ROS "AUTO" CACHE STRING "Build the camera_ros target: AUTO, ON, or OFF")
set_property(CACHE MJ_KDL_WITH_ROS PROPERTY STRINGS AUTO ON OFF)
set(MJ_KDL_ROS_ENABLED OFF)
if(NOT MJ_KDL_WITH_ROS STREQUAL "OFF")
    set(_mj_kdl_ros_packages rclcpp sensor_msgs image_transport rosgraph_msgs)
    set(MJ_KDL_ROS_ENABLED ON)
    foreach(_package IN LISTS _mj_kdl_ros_packages)
        if(MJ_KDL_WITH_ROS STREQUAL "AUTO")
            find_package(${_package} QUIET)
        else()
            find_package(${_package} REQUIRED)
        endif()
        if(NOT ${_package}_FOUND)
            set(MJ_KDL_ROS_ENABLED OFF)
        endif()
    endforeach()
endif()
if(MJ_KDL_ROS_ENABLED)
    add_library(mj_kdl_camera_ros SHARED src/camera_ros.cpp)
    add_library(mj_kdl_wrapper::camera_ros ALIAS mj_kdl_camera_ros)
    set_target_properties(mj_kdl_camera_ros PROPERTIES EXPORT_NAME camera_ros)
    target_link_libraries(mj_kdl_camera_ros PUBLIC
      mj_kdl_wrapper
      rclcpp::rclcpp
      image_transport::image_transport
      ${sensor_msgs_TARGETS}
      ${rosgraph_msgs_TARGETS}
    )
    # image_transport 6 (Kilted and later) takes node interfaces and an rclcpp::QoS, and
    # deprecates the rclcpp::Node * / rmw_qos_profile_t overload that 5 (Jazzy) only has.
    if(image_transport_VERSION VERSION_GREATER_EQUAL 6)
        target_compile_definitions(mj_kdl_camera_ros PRIVATE MJ_KDL_IMAGE_TRANSPORT_QOS)
    endif()
    target_compile_options(mj_kdl_camera_ros PRIVATE -Wall -Wextra -Wpedantic)
    message(STATUS "ROS    : camera_ros enabled (mj_kdl_wrapper::camera_ros)")
else()
    message(STATUS "ROS    : rclcpp/sensor_msgs/image_transport/rosgraph_msgs not found; "
      "camera_ros not built")
endif()

option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
if(BUILD_PYTHON_BINDINGS)
    # The wheel does not bundle libmujoco; it reuses the shared library shipped
    # by the pinned `mujoco` pip package, which installs as a sibling top-level
    # package (site-packages/mujoco). $ORIGIN/../mujoco resolves to it.
    set_target_properties(mj_kdl_wrapper PROPERTIES
      BUILD_RPATH "${MUJOCO_LIB_DIR}"
      INSTALL_RPATH "$ORIGIN;$ORIGIN/../mujoco"
    )

    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        execute_process(
          COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
          OUTPUT_VARIABLE _pybind11_cmake_dir
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET
        )
        if(_pybind11_cmake_dir)
            list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmake_dir}")
        endif()
        find_package(pybind11 CONFIG REQUIRED)
    endif()

    # PyKDL and liborocos-kdl are built and bundled only when this build owns the
    # KDL (ExternalProject). When KDL is consumed from elsewhere (a shared prefix
    # or a found package), its PyKDL comes from there too, so we neither build nor
    # bundle them - the goal is exactly one copy across the workspace/process.
    if(_MJ_KDL_BUILD_OROCOS_KDL)
        # Build PyKDL from the same checkout, at build time, against the
        # ExternalProject-installed Orocos KDL. python_orocos_kdl is a separate CMake
        # project that find_package()s an installed orocos_kdl, so it must run after
        # orocos_kdl_ep has installed (enforced via DEPENDS).
        set(_MJ_KDL_PYKDL_INSTALL "${CMAKE_BINARY_DIR}/python_orocos_kdl_install")
        ExternalProject_Add(pykdl_ep
          DEPENDS          orocos_kdl_ep
          SOURCE_DIR       "${_MJ_KDL_OROCOS_SRC}"
          SOURCE_SUBDIR    python_orocos_kdl
          DOWNLOAD_COMMAND ""
          CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX=${_MJ_KDL_PYKDL_INSTALL}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
            -DCMAKE_PREFIX_PATH=${_MJ_KDL_OROCOS_PREFIX}
            -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
            -Dpybind11_DIR=${pybind11_DIR}
            -DPYTHON_SITE_PACKAGES_INSTALL_DIR=.
            "-DCMAKE_INSTALL_RPATH=\$ORIGIN/mj_kdl_wrapper"
            # DT_RPATH, as for the wrapper: LD_LIBRARY_PATH must not swap in a system KDL.
            -DCMAKE_MODULE_LINKER_FLAGS=-Wl,--disable-new-dtags
            -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--disable-new-dtags
          UPDATE_DISCONNECTED TRUE
        )
        # Bundle PyKDL (top-level module) and the Orocos KDL shared library from the
        # ExternalProject install trees. FILES_MATCHING globs at install time, after
        # both ExternalProject builds have run.
        install(DIRECTORY "${_MJ_KDL_PYKDL_INSTALL}/"
          DESTINATION .
          FILES_MATCHING PATTERN "PyKDL*.so"
        )
        install(FILES licenses/LGPL-2.1.txt licenses/KDL_SOURCE.txt
          DESTINATION .
        )
        install(DIRECTORY "${_MJ_KDL_OROCOS_PREFIX}/lib/"
          DESTINATION mj_kdl_wrapper
          FILES_MATCHING PATTERN "liborocos-kdl.so*"
        )
        install(FILES licenses/LGPL-2.1.txt licenses/KDL_SOURCE.txt
          DESTINATION mj_kdl_wrapper
        )
    endif()

    pybind11_add_module(_mj_kdl_wrapper
      python/src/bindings.cpp
    )
    target_link_libraries(_mj_kdl_wrapper PRIVATE mj_kdl_wrapper)
    # Python views and records through the same C++ viewer and recorder, not a second UI.
    foreach(_component render viewer)
        if(TARGET mj_kdl_${_component})
            string(TOUPPER "${_component}" _component_upper)
            target_link_libraries(_mj_kdl_wrapper PRIVATE mj_kdl_${_component})
            target_compile_definitions(_mj_kdl_wrapper PRIVATE
              MJ_KDL_PYTHON_${_component_upper})
            set_target_properties(mj_kdl_${_component} PROPERTIES
              INSTALL_RPATH "$ORIGIN;$ORIGIN/../mujoco")
            install(TARGETS mj_kdl_${_component}
              LIBRARY DESTINATION mj_kdl_wrapper
              RUNTIME DESTINATION mj_kdl_wrapper
            )
        endif()
    endforeach()
    if(TARGET mj_kdl_render)
        install(DIRECTORY "${MJ_KDL_RENDER_ASSETS_DIR}" DESTINATION mj_kdl_wrapper)
    endif()
    if(TARGET orocos_kdl_ep)
        # bindings.cpp includes KDL headers produced by the ExternalProject.
        add_dependencies(_mj_kdl_wrapper orocos_kdl_ep)
    endif()
    target_compile_features(_mj_kdl_wrapper PRIVATE cxx_std_20)
    target_compile_options(_mj_kdl_wrapper PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_definitions(_mj_kdl_wrapper PRIVATE
      MJ_KDL_WRAPPER_VERSION="${PROJECT_VERSION}"
    )
    # The extension calls MuJoCo directly, so libmujoco is a direct dependency
    # of this module too. DT_RUNPATH is not transitive, so it needs its own
    # entry for the sibling mujoco pip package alongside $ORIGIN.
    set_target_properties(_mj_kdl_wrapper PROPERTIES
      BUILD_RPATH "$<TARGET_FILE_DIR:mj_kdl_wrapper>"
      INSTALL_RPATH "$ORIGIN;$ORIGIN/../mujoco"
    )
    install(TARGETS _mj_kdl_wrapper
      LIBRARY DESTINATION mj_kdl_wrapper
      RUNTIME DESTINATION mj_kdl_wrapper
    )
    install(TARGETS mj_kdl_wrapper
      LIBRARY DESTINATION mj_kdl_wrapper
      RUNTIME DESTINATION mj_kdl_wrapper
    )
    install(FILES
      NOTICE
      THIRD_PARTY_LICENSES.md
      licenses/Apache-2.0.txt
      licenses/Robotiq-2F85-BSD-2-Clause.txt
      licenses/Kinova-Gen3-BSD-3-Clause.txt
      DESTINATION mj_kdl_wrapper
    )
    # Installed, not mapped in by pyproject: a wheel mapping above this package leaks sys.path.
    install(DIRECTORY assets/
      DESTINATION mj_kdl_wrapper/assets
    )
    # libmujoco and its plugins are not bundled: the wheel depends on the matching mujoco pip
    # package, which loads them, and links its library at runtime via $ORIGIN/../mujoco. The
    # render and viewer libraries carry Filament and Studio statically, with their assets beside.
endif()

# Examples (GUI by default; pass --headless to skip the window)
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES AND MJ_KDL_VIEWER_ENABLED)
    add_subdirectory(src/examples)
elseif(BUILD_EXAMPLES)
    message(STATUS "examples: not built, they open the viewer (MJ_KDL_WITH_VIEWER)")
endif()

# Tests
option(BUILD_TESTS "Build and register tests with CTest" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Documentation (Doxygen + local CSS)
# Build separately:  cmake --build build --target docs
# Requires:  sudo apt install doxygen
option(BUILD_DOCS "Generate HTML documentation with Doxygen" OFF)
if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)

    set(MJ_KDL_KDL_DOXYGEN_INPUT_DIRS)
    set(_mj_kdl_kdl_tag_depends)
    if(_MJ_KDL_BUILD_OROCOS_KDL)
        # The fork's headers exist only once orocos_kdl_ep has installed them.
        set(MJ_KDL_KDL_DOXYGEN_INPUT_DIRS "${_MJ_KDL_OROCOS_PREFIX}/include/kdl")
        set(_mj_kdl_kdl_tag_depends orocos_kdl_ep)
    else()
        foreach(_kdl_inc IN LISTS orocos_kdl_INCLUDE_DIRS)
            if(EXISTS "${_kdl_inc}/kdl/chain.hpp")
                list(APPEND MJ_KDL_KDL_DOXYGEN_INPUT_DIRS "${_kdl_inc}/kdl")
            endif()
        endforeach()
    endif()

    set(MJ_KDL_DOCS_DEPENDS)
    set(MJ_KDL_DOXYGEN_TAGFILES)
    if(MJ_KDL_KDL_DOXYGEN_INPUT_DIRS)
        list(REMOVE_DUPLICATES MJ_KDL_KDL_DOXYGEN_INPUT_DIRS)
        string(REPLACE ";" " " MJ_KDL_KDL_DOXYGEN_INPUTS "${MJ_KDL_KDL_DOXYGEN_INPUT_DIRS}")
        set(MJ_KDL_KDL_TAG_FILE "${CMAKE_CURRENT_BINARY_DIR}/docs/kdl.tag")
        configure_file(
          ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/Doxyfile.kdl.in
          ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.kdl
          @ONLY
        )
        add_custom_command(
          OUTPUT ${MJ_KDL_KDL_TAG_FILE}
          # Doxygen creates only the last level of HTML_OUTPUT (html/kdl).
          COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/docs/html
          COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.kdl
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.kdl ${_mj_kdl_kdl_tag_depends}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          COMMENT "Generating Orocos KDL Doxygen tag file..."
          VERBATIM
        )
        list(APPEND MJ_KDL_DOCS_DEPENDS ${MJ_KDL_KDL_TAG_FILE})
        set(MJ_KDL_DOXYGEN_TAGFILES "${MJ_KDL_KDL_TAG_FILE}=kdl/")
        message(STATUS "Doxygen KDL tag input: ${MJ_KDL_KDL_DOXYGEN_INPUT_DIRS}")
    else()
        message(WARNING
          "Doxygen: KDL headers were not found; external KDL symbols will not be resolved")
    endif()

    configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    @ONLY
  )

    add_custom_target(docs
    DEPENDS ${MJ_KDL_DOCS_DEPENDS}
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_CURRENT_SOURCE_DIR}/docs/screenshots
      ${CMAKE_CURRENT_BINARY_DIR}/docs/html/docs/screenshots
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen..."
    VERBATIM
  )
    message(STATUS "Doxygen: docs target enabled (run: cmake --build build --target docs)")
endif()

# Install
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_MJ_KDL_INSTALL_CPP_PACKAGE_DEFAULT ON)
if(DEFINED SKBUILD)
  set(_MJ_KDL_INSTALL_CPP_PACKAGE_DEFAULT OFF)
endif()
option(
  MJ_KDL_INSTALL_CPP_PACKAGE
  "Install the C++ library, headers, and CMake package config"
  ${_MJ_KDL_INSTALL_CPP_PACKAGE_DEFAULT}
)
# Make the install self-contained: also install the Orocos KDL fork we built (its
# library, headers, and CMake package config) into the same prefix, so the
# installed wrapper loads KDL via $ORIGIN and sibling packages in the prefix can
# find_package(orocos_kdl). Disable to keep KDL out of a shared prefix (e.g.
# /usr/local) where it could shadow a distro KDL. Has no effect when consuming an
# external KDL via MJ_KDL_OROCOS_KDL_INSTALL_DIR (that prefix is already shared).
option(
  MJ_KDL_INSTALL_BUNDLED_KDL
  "Install the built Orocos KDL fork into the prefix so the install is self-contained"
  ON
)

if(MJ_KDL_INSTALL_CPP_PACKAGE)
    install(TARGETS mj_kdl_wrapper
      EXPORT mj_kdl_wrapperTargets
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    foreach(_component mj_kdl_render mj_kdl_viewer mj_kdl_camera_ros)
        if(TARGET ${_component})
            install(TARGETS ${_component}
              EXPORT mj_kdl_wrapperTargets
              RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
              LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
              ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            )
        endif()
    endforeach()
    # MuJoCo is built here, so the prefix carries it: its library, headers and plugins, and
    # the Filament and Studio assets the render and viewer libraries read beside them.
    install(FILES "${_MJ_KDL_MUJOCO_LIB}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(CODE "file(CREATE_LINK libmujoco.so.${MJ_KDL_MUJOCO_VERSION}
      \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libmujoco.so\" SYMBOLIC)")
    install(DIRECTORY "${MJ_KDL_MUJOCO_DIR}/include/mujoco"
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
      FILES_MATCHING PATTERN "*.h"
    )
    install(DIRECTORY "${MJ_KDL_MUJOCO_PLUGIN_DIR}"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
      FILES_MATCHING PATTERN "*.so"
    )
    if(MJ_KDL_RENDER_ENABLED)
        install(DIRECTORY "${MJ_KDL_RENDER_ASSETS_DIR}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()
    # The C++ install ships no robot assets, so only the licenses of what it compiles in.
    install(FILES
      LICENSE
      NOTICE
      THIRD_PARTY_LICENSES.md
      licenses/Apache-2.0.txt
      DESTINATION ${CMAKE_INSTALL_DOCDIR}
    )
    # Consumers read the bundled models from the user cache, not the prefix.
    if(_MJ_KDL_CACHE_ROOT)
        install(DIRECTORY assets/ DESTINATION "${_MJ_KDL_CACHE_ROOT}/assets")
        install(FILES
          licenses/Robotiq-2F85-BSD-2-Clause.txt
          licenses/Kinova-Gen3-BSD-3-Clause.txt
          DESTINATION "${_MJ_KDL_CACHE_ROOT}/assets"
        )
    endif()
    install(FILES include/mj_kdl_wrapper/mj_kdl_wrapper.hpp
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mj_kdl_wrapper
    )
    foreach(_header render viewer camera_ros)
        if(TARGET mj_kdl_wrapper::${_header})
            install(FILES include/mj_kdl_wrapper/${_header}.hpp
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mj_kdl_wrapper
            )
        endif()
    endforeach()
    if(TARGET mj_kdl_wrapper::camera_ros)
        install(FILES include/mj_kdl_wrapper/clock_ros.hpp
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mj_kdl_wrapper
        )
    endif()

    # libmujoco sits next to the libraries; KDL is either bundled ($ORIGIN, below) or loaded
    # from its own prefix.
    foreach(_lib mj_kdl_wrapper mj_kdl_render mj_kdl_viewer)
        if(TARGET ${_lib})
            set_property(TARGET ${_lib} APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
        endif()
    endforeach()
    set(_mj_kdl_kdl_dir "")
    if(NOT _MJ_KDL_BUILD_OROCOS_KDL)
        get_target_property(_mj_kdl_kdl_type orocos-kdl TYPE)
        if(_mj_kdl_kdl_type MATCHES "SHARED_LIBRARY|UNKNOWN_LIBRARY")
            get_target_property(_mj_kdl_kdl_location orocos-kdl LOCATION)
            get_filename_component(_mj_kdl_kdl_dir "${_mj_kdl_kdl_location}" DIRECTORY)
            set_property(TARGET mj_kdl_wrapper APPEND PROPERTY INSTALL_RPATH "${_mj_kdl_kdl_dir}")
        endif()
    elseif(NOT MJ_KDL_INSTALL_BUNDLED_KDL)
        message(WARNING
          "MJ_KDL_INSTALL_BUNDLED_KDL=OFF: the installed library has no path to the KDL fork; "
          "the loader then takes the first liborocos-kdl it finds, possibly a system one.")
    endif()
    if(TARGET mj_kdl_camera_ros)
        set_property(TARGET mj_kdl_camera_ros APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
        if(_mj_kdl_kdl_dir)
            set_property(TARGET mj_kdl_camera_ros APPEND PROPERTY
              INSTALL_RPATH "${_mj_kdl_kdl_dir}")
        endif()
    endif()
    # Self-contained install: bundle the fork we built (lib + headers + its CMake
    # package config) into this prefix and rpath the wrapper to $ORIGIN so it loads
    # the co-installed liborocos-kdl. Skipped when consuming an external KDL, which
    # is already shared from its own prefix.
    set(_MJ_KDL_KDL_IN_PREFIX OFF)
    if(MJ_KDL_INSTALL_BUNDLED_KDL AND _MJ_KDL_BUILD_OROCOS_KDL)
        set(_MJ_KDL_KDL_IN_PREFIX ON)
        install(DIRECTORY "${_MJ_KDL_OROCOS_PREFIX}/"
          DESTINATION .
          USE_SOURCE_PERMISSIONS
        )
        install(FILES licenses/LGPL-2.1.txt licenses/KDL_SOURCE.txt
          DESTINATION ${CMAKE_INSTALL_DOCDIR}
        )
        set_property(TARGET mj_kdl_wrapper APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
    endif()

    # KDL flags come from the fork's orocos-kdl.pc; an external KDL also needs its rpath here.
    file(RELATIVE_PATH _MJ_KDL_PC_PREFIX
      "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig" "${CMAKE_INSTALL_PREFIX}")
    string(REGEX REPLACE "/$" "" _MJ_KDL_PC_PREFIX "${_MJ_KDL_PC_PREFIX}")
    set(_MJ_KDL_PC_KDL_RPATH "")
    if(_mj_kdl_kdl_dir)
        set(_MJ_KDL_PC_KDL_RPATH "-Wl,-rpath,${_mj_kdl_kdl_dir}")
    endif()
    configure_file(cmake/mj_kdl_wrapper.pc.in
      ${CMAKE_CURRENT_BINARY_DIR}/mj_kdl_wrapper.pc @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mj_kdl_wrapper.pc
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    )

    install(EXPORT mj_kdl_wrapperTargets
      FILE mj_kdl_wrapperTargets.cmake
      NAMESPACE mj_kdl_wrapper::
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mj_kdl_wrapper
    )

    configure_package_config_file(
      cmake/mj_kdl_wrapperConfig.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/mj_kdl_wrapperConfig.cmake
      INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mj_kdl_wrapper
    )

    write_basic_package_version_file(
      ${CMAKE_CURRENT_BINARY_DIR}/mj_kdl_wrapperConfigVersion.cmake
      VERSION ${PROJECT_VERSION}
      COMPATIBILITY SameMinorVersion
    )

    install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/mj_kdl_wrapperConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/mj_kdl_wrapperConfigVersion.cmake
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mj_kdl_wrapper
    )
endif()
