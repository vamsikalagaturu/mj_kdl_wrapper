cmake_minimum_required(VERSION 3.16)

include(cmake/Versions.cmake)

project(mjkdl VERSION ${MJKDL_VERSION} LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# The downloaded MuJoCo release and the installed bundled models are cached here.
if(DEFINED ENV{XDG_CACHE_HOME} AND NOT "$ENV{XDG_CACHE_HOME}" STREQUAL "")
    set(_MJKDL_CACHE_ROOT "$ENV{XDG_CACHE_HOME}/mjkdl")
elseif(DEFINED ENV{HOME} AND NOT "$ENV{HOME}" STREQUAL "")
    set(_MJKDL_CACHE_ROOT "$ENV{HOME}/.cache/mjkdl")
endif()
set(MJKDL_MUJOCO_DEFAULT_DIR "")
if(_MJKDL_CACHE_ROOT)
    set(MJKDL_MUJOCO_DEFAULT_DIR "${_MJKDL_CACHE_ROOT}/mujoco-${MJKDL_MUJOCO_VERSION}")
endif()

# MuJoCo: downloaded into the cache by default. Point MJKDL_MUJOCO_DIR at an
# existing install to use that instead (no system paths are searched).
option(
  MJKDL_FETCH_MUJOCO
  "Download the supported MuJoCo release if MJKDL_MUJOCO_DIR is not available"
  ON
)
set(MJKDL_MUJOCO_URL "" CACHE STRING "Override MuJoCo release archive URL")

# Orocos KDL is always the secorolab fork built from source - it is the only KDL this
# project uses; no system liborocos-kdl is ever consulted. `vcs import <
# mjkdl.repos` checks it out at the default location.
set(
  MJKDL_OROCOS_KDL_DIR "${CMAKE_SOURCE_DIR}/third_party/orocos_kinematics_dynamics"
  CACHE PATH
  "Orocos KDL fork source checkout"
)
# Consume an already-installed Orocos KDL prefix instead of building the fork.
# Intended for ROS 2 / colcon, where the secorolab fork is a workspace package
# and a single shared liborocos-kdl (and PyKDL) must be used everywhere to avoid
# loading two copies with the same SONAME in one process. When set, no
# ExternalProject build runs and nothing is bundled; the prefix must contain the
# secorolab fork. Takes precedence over MJKDL_OROCOS_KDL_DIR.
set(
  MJKDL_OROCOS_KDL_INSTALL_DIR ""
  CACHE PATH
  "Pre-installed Orocos KDL prefix to consume (skips building/bundling the fork)"
)
# Consume an Orocos KDL provided as a CMake package on CMAKE_PREFIX_PATH via
# find_package(orocos_kdl). This is the natural way to share one KDL across a
# workspace - a colcon overlay (the fork built as the orocos_kdl package) or any
# prefix on CMAKE_PREFIX_PATH where the secorolab fork is installed. When ON, no
# ExternalProject build runs and nothing is bundled. The found package MUST be the
# secorolab fork, not a stock distro KDL. Takes precedence over
# MJKDL_OROCOS_KDL_INSTALL_DIR and the in-tree build.
option(
  MJKDL_OROCOS_KDL_FROM_PACKAGE
  "Consume Orocos KDL via find_package(orocos_kdl) instead of building the fork"
  OFF
)

# MuJoCo - default to the user cache, override with -DMJKDL_MUJOCO_DIR=...
if(NOT DEFINED CACHE{MJKDL_MUJOCO_DIR} AND NOT MJKDL_MUJOCO_DEFAULT_DIR)
    message(FATAL_ERROR "Set XDG_CACHE_HOME or HOME for the default MuJoCo cache, "
      "or pass MJKDL_MUJOCO_DIR explicitly.")
endif()
set(MJKDL_MUJOCO_DIR "${MJKDL_MUJOCO_DEFAULT_DIR}" CACHE PATH "MuJoCo install location")
if(MJKDL_FETCH_MUJOCO AND NOT EXISTS "${MJKDL_MUJOCO_DIR}/include/mujoco/mujoco.h")
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR
          "Automatic MuJoCo download is currently supported on Linux only. "
          "Set MJKDL_MUJOCO_DIR to an existing MuJoCo ${MJKDL_MUJOCO_VERSION} install.")
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
        set(_mjkdl_mujoco_platform "linux-x86_64")
    else()
        message(FATAL_ERROR
          "Automatic MuJoCo download does not support ${CMAKE_SYSTEM_PROCESSOR}. "
          "Set MJKDL_MUJOCO_DIR to an existing MuJoCo ${MJKDL_MUJOCO_VERSION} install.")
    endif()
    set(
      _mjkdl_mujoco_default_url
      "https://github.com/google-deepmind/mujoco/releases/download/${MJKDL_MUJOCO_VERSION}/mujoco-${MJKDL_MUJOCO_VERSION}-${_mjkdl_mujoco_platform}.tar.gz"
    )
    if(NOT MJKDL_MUJOCO_URL)
        set(
          MJKDL_MUJOCO_URL
          "${_mjkdl_mujoco_default_url}"
          CACHE STRING
          "Override MuJoCo release archive URL"
          FORCE
        )
    endif()
    # Only the default archive has a known hash; an overridden URL is the user's to vouch for.
    set(_mjkdl_mujoco_hash)
    if(MJKDL_MUJOCO_URL STREQUAL _mjkdl_mujoco_default_url)
        set(_mjkdl_mujoco_hash URL_HASH "SHA256=${MJKDL_MUJOCO_SHA256}")
    endif()
    include(FetchContent)
    if(POLICY CMP0135)
      cmake_policy(SET CMP0135 NEW)
    endif()
    message(STATUS "MuJoCo : fetching ${MJKDL_MUJOCO_URL}")
    # Download-only: SOURCE_SUBDIR names a path with no CMakeLists.txt, so
    # FetchContent_MakeAvailable populates the archive without add_subdirectory.
    FetchContent_Declare(
      mujoco_release
      URL "${MJKDL_MUJOCO_URL}"
      ${_mjkdl_mujoco_hash}
      SOURCE_DIR "${MJKDL_MUJOCO_DIR}"
      SOURCE_SUBDIR _mjkdl_no_configure
    )
    FetchContent_MakeAvailable(mujoco_release)
    set(MJKDL_MUJOCO_DIR "${mujoco_release_SOURCE_DIR}" CACHE PATH "MuJoCo install location" FORCE)
endif()
if(NOT EXISTS "${MJKDL_MUJOCO_DIR}/include/mujoco/mujoco.h")
    message(FATAL_ERROR
      "MuJoCo ${MJKDL_MUJOCO_VERSION} was not found at ${MJKDL_MUJOCO_DIR}. "
      "Set MJKDL_MUJOCO_DIR to an existing install, or set MJKDL_FETCH_MUJOCO=ON "
      "to download it into the cache.")
endif()
find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
          PATHS "${MJKDL_MUJOCO_DIR}/include"
          NO_DEFAULT_PATH REQUIRED)
file(STRINGS "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h" MUJOCO_VERSION_HEADER_LINE
     REGEX "^#define mjVERSION_HEADER ")
string(REGEX REPLACE ".*mjVERSION_HEADER +([0-9]+).*" "\\1" MUJOCO_VERSION_HEADER
       "${MUJOCO_VERSION_HEADER_LINE}")
if(NOT MUJOCO_VERSION_HEADER STREQUAL "${MJKDL_MUJOCO_VERSION_HEADER}")
    message(FATAL_ERROR
      "mjkdl requires MuJoCo ${MJKDL_MUJOCO_VERSION} "
      "(mjVERSION_HEADER ${MJKDL_MUJOCO_VERSION_HEADER}), "
      "but found ${MUJOCO_VERSION_HEADER} at ${MUJOCO_INCLUDE_DIR}")
endif()
find_library(MUJOCO_LIBRARY mujoco
             PATHS "${MJKDL_MUJOCO_DIR}/lib"
             NO_DEFAULT_PATH REQUIRED)
get_filename_component(MUJOCO_LIB_DIR "${MUJOCO_LIBRARY}" DIRECTORY)
list(APPEND CMAKE_BUILD_RPATH "${MUJOCO_LIB_DIR}")
message(STATUS "MuJoCo : ${MUJOCO_LIBRARY}")

# For consumers wanting MuJoCo without the KDL, glfw and OpenGL that mjkdl brings.
if(NOT TARGET mujoco::mujoco)
    add_library(mujoco::mujoco SHARED IMPORTED GLOBAL)
    set_target_properties(mujoco::mujoco PROPERTIES
      IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}")
endif()

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

# EGL for headless video recording (VideoRecorder API)
option(BUILD_RECORDER "Build headless EGL/ffmpeg video recorder support" ON)
if(BUILD_RECORDER)
    find_library(EGL_LIBRARY EGL)
    if(EGL_LIBRARY)
        message(STATUS "EGL    : ${EGL_LIBRARY} (VideoRecorder enabled)")
    else()
        message(WARNING "EGL library not found; VideoRecorder will be a no-op stub")
        set(BUILD_RECORDER OFF)
    endif()
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
set(_MJKDL_BUILD_OROCOS_KDL FALSE)
if(MJKDL_OROCOS_KDL_FROM_PACKAGE)
  # Consume an Orocos KDL provided as a CMake package on CMAKE_PREFIX_PATH (a
  # colcon overlay, or any prefix where the fork is installed). find_package
  # creates the orocos-kdl imported target and sets orocos_kdl_INCLUDE_DIRS.
  find_package(orocos_kdl REQUIRED)
  set(orocos_kdl_LIBRARIES orocos-kdl)
  set(orocos_kdl_DIR "found: orocos_kdl package")
  message(STATUS "Orocos KDL: using found orocos_kdl package")
elseif(MJKDL_OROCOS_KDL_INSTALL_DIR)
  # Consume a pre-installed Orocos KDL (e.g. a colcon workspace overlay). No
  # ExternalProject build, no bundling: a single shared liborocos-kdl is used by
  # the wrapper and by any other package in the same workspace/process.
  find_path(MJKDL_OROCOS_KDL_INCLUDE_DIR kdl/chain.hpp
            PATHS "${MJKDL_OROCOS_KDL_INSTALL_DIR}/include"
            NO_DEFAULT_PATH REQUIRED)
  find_library(MJKDL_OROCOS_KDL_LIBRARY orocos-kdl
               PATHS "${MJKDL_OROCOS_KDL_INSTALL_DIR}/lib"
               NO_DEFAULT_PATH REQUIRED)
  add_library(orocos-kdl SHARED IMPORTED GLOBAL)
  set_target_properties(orocos-kdl PROPERTIES
    IMPORTED_LOCATION "${MJKDL_OROCOS_KDL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MJKDL_OROCOS_KDL_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES Eigen3::Eigen
  )
  get_target_property(_mjkdl_eigen_inc Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
  set(orocos_kdl_LIBRARIES    orocos-kdl)
  set(orocos_kdl_INCLUDE_DIRS "${MJKDL_OROCOS_KDL_INCLUDE_DIR}" ${_mjkdl_eigen_inc})
  set(orocos_kdl_DIR          "installed: ${MJKDL_OROCOS_KDL_INSTALL_DIR}")
  message(STATUS "Orocos KDL: using installed prefix ${MJKDL_OROCOS_KDL_INSTALL_DIR}")
else()
set(_MJKDL_BUILD_OROCOS_KDL TRUE)
set(_MJKDL_OROCOS_PREFIX "${CMAKE_BINARY_DIR}/orocos_kdl_install")
set(_MJKDL_OROCOS_SRC "${MJKDL_OROCOS_KDL_DIR}")
if(NOT EXISTS "${_MJKDL_OROCOS_SRC}/orocos_kdl/CMakeLists.txt")
  message(FATAL_ERROR
    "No Orocos KDL source at ${_MJKDL_OROCOS_SRC}. Run "
    "'vcs import < mjkdl.repos', point MJKDL_OROCOS_KDL_DIR at a "
    "fork checkout, or set MJKDL_OROCOS_KDL_INSTALL_DIR. No system KDL is used.")
endif()
message(STATUS "Orocos KDL: building ${_MJKDL_OROCOS_SRC}")
ExternalProject_Add(orocos_kdl_ep
  DOWNLOAD_COMMAND ""
  SOURCE_DIR     "${_MJKDL_OROCOS_SRC}"
  SOURCE_SUBDIR  orocos_kdl
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${_MJKDL_OROCOS_PREFIX}
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
  BUILD_BYPRODUCTS "${_MJKDL_OROCOS_PREFIX}/lib/liborocos-kdl.so"
)
# IMPORTED target the wrapper links against. The install tree is produced by the
# ExternalProject build step; create the include dir up front so the path
# recorded on the target is valid at configure time.
file(MAKE_DIRECTORY "${_MJKDL_OROCOS_PREFIX}/include")
add_library(orocos-kdl SHARED IMPORTED GLOBAL)
set_target_properties(orocos-kdl PROPERTIES
  IMPORTED_LOCATION "${_MJKDL_OROCOS_PREFIX}/lib/liborocos-kdl.so"
  INTERFACE_INCLUDE_DIRECTORIES "${_MJKDL_OROCOS_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES Eigen3::Eigen
)
get_target_property(_mjkdl_eigen_inc Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
set(orocos_kdl_LIBRARIES    orocos-kdl)
set(orocos_kdl_INCLUDE_DIRS "${_MJKDL_OROCOS_PREFIX}/include" ${_mjkdl_eigen_inc})
set(orocos_kdl_DIR          "secorolab fork (ExternalProject)")
endif()

message(STATUS "KDL    : ${orocos_kdl_LIBRARIES}")
message(STATUS "KDL inc: ${orocos_kdl_INCLUDE_DIRS}")
message(STATUS "KDL cfg: ${orocos_kdl_DIR}")


# ffmpeg-backed PNG writer used by the local Simulate UI screenshot path.
set(MJ_SIM_DIR "${MJKDL_MUJOCO_DIR}/simulate")
set(MJ_SIM_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/simulate_ui")
set(SIMULATE_UI_SOURCES
  ${MJ_SIM_LOCAL_DIR}/simulate.cc
  ${MJ_SIM_DIR}/glfw_adapter.cc
  ${MJ_SIM_DIR}/glfw_dispatch.cc
  ${MJ_SIM_DIR}/platform_ui_adapter.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/image_io.cpp
)
message(STATUS "image_io: using ffmpeg-backed PNG writer for UI screenshots")

# Optional viewer UI sections (hidden by default)
option(SHOW_EQUALITY_PANEL "Show the Equality UI section in the viewer" OFF)
option(SHOW_GROUP_PANEL    "Show the Group-enable UI section in the viewer" OFF)

add_library(mujoco_simulate_ui STATIC ${SIMULATE_UI_SOURCES})
target_include_directories(mujoco_simulate_ui PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MJ_SIM_LOCAL_DIR}
  ${MJ_SIM_DIR}
  ${MUJOCO_INCLUDE_DIR}
)
target_link_libraries(mujoco_simulate_ui PUBLIC
  mujoco::mujoco
  glfw
  OpenGL::GL
)
target_compile_options(mujoco_simulate_ui PRIVATE -w)
if(SHOW_EQUALITY_PANEL)
  target_compile_definitions(mujoco_simulate_ui PRIVATE MJKDL_SHOW_EQUALITY)
endif()
if(SHOW_GROUP_PANEL)
  target_compile_definitions(mujoco_simulate_ui PRIVATE MJKDL_SHOW_GROUP)
endif()
set_target_properties(mujoco_simulate_ui PROPERTIES
  CXX_STANDARD 20 POSITION_INDEPENDENT_CODE ON)

# Library
add_library(mjkdl SHARED
  src/mjkdl.cpp
)

target_include_directories(mjkdl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(mjkdl PRIVATE
  ${MJ_SIM_LOCAL_DIR}
  ${MJ_SIM_DIR}
)

target_link_libraries(mjkdl PUBLIC
  mujoco::mujoco
  glfw
  OpenGL::GL
  ${orocos_kdl_LIBRARIES}
  # The installed fork's orocos-kdl target does not carry Eigen.
  Eigen3::Eigen
)
target_link_libraries(mjkdl PRIVATE
  mujoco_simulate_ui
)
# DT_RPATH, not RUNPATH: LD_LIBRARY_PATH must not swap the KDL fork for a system liborocos-kdl
# with the same soname.
target_link_options(mjkdl PUBLIC "LINKER:--disable-new-dtags")

# When KDL is built via ExternalProject, its headers and library only exist
# after that build step. Order it before the wrapper compiles/links.
if(TARGET orocos_kdl_ep)
    add_dependencies(mjkdl orocos_kdl_ep)
endif()

target_compile_options(mjkdl PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(mjkdl PRIVATE
  MUJOCO_PLUGIN_DIR="${MJKDL_MUJOCO_DIR}/bin/mujoco_plugin"
)
# dladdr() finds a mujoco_plugin directory installed next to the library.
target_link_libraries(mjkdl PRIVATE ${CMAKE_DL_LIBS})
if(BUILD_RECORDER AND EGL_LIBRARY)
    target_link_libraries(mjkdl PRIVATE ${EGL_LIBRARY})
    target_compile_definitions(mjkdl PRIVATE MJKDL_HAS_EGL)
endif()

# Optional ROS component: publishes an already-rendered camera frame as sensor_msgs/Image.
# AUTO builds it when rclcpp and sensor_msgs are both findable; a ROS-less configure simply
# leaves the target out, and the core library is unchanged either way.
set(MJKDL_WITH_ROS "AUTO" CACHE STRING "Build the camera_ros target: AUTO, ON, or OFF")
set_property(CACHE MJKDL_WITH_ROS PROPERTY STRINGS AUTO ON OFF)
set(MJKDL_ROS_ENABLED OFF)
if(NOT MJKDL_WITH_ROS STREQUAL "OFF")
    if(MJKDL_WITH_ROS STREQUAL "AUTO")
        find_package(rclcpp QUIET)
        find_package(sensor_msgs QUIET)
    else()
        find_package(rclcpp REQUIRED)
        find_package(sensor_msgs REQUIRED)
    endif()
    if(rclcpp_FOUND AND sensor_msgs_FOUND)
        set(MJKDL_ROS_ENABLED ON)
    endif()
endif()
if(MJKDL_ROS_ENABLED)
    add_library(mjkdl_camera_ros SHARED src/camera_ros.cpp)
    add_library(mjkdl::camera_ros ALIAS mjkdl_camera_ros)
    set_target_properties(mjkdl_camera_ros PROPERTIES EXPORT_NAME camera_ros)
    target_link_libraries(mjkdl_camera_ros PUBLIC
      mjkdl
      rclcpp::rclcpp
      ${sensor_msgs_TARGETS}
    )
    target_compile_options(mjkdl_camera_ros PRIVATE -Wall -Wextra -Wpedantic)
    message(STATUS "ROS    : camera_ros enabled (mjkdl::camera_ros)")
else()
    message(STATUS "ROS    : rclcpp/sensor_msgs not found; camera_ros not built")
endif()

option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
if(BUILD_PYTHON_BINDINGS)
    # The wheel does not bundle libmujoco; it reuses the shared library shipped
    # by the pinned `mujoco` pip package, which installs as a sibling top-level
    # package (site-packages/mujoco). $ORIGIN/../mujoco resolves to it.
    set_target_properties(mjkdl PROPERTIES
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
    if(_MJKDL_BUILD_OROCOS_KDL)
        # Build PyKDL from the same checkout, at build time, against the
        # ExternalProject-installed Orocos KDL. python_orocos_kdl is a separate CMake
        # project that find_package()s an installed orocos_kdl, so it must run after
        # orocos_kdl_ep has installed (enforced via DEPENDS).
        set(_MJKDL_PYKDL_INSTALL "${CMAKE_BINARY_DIR}/python_orocos_kdl_install")
        ExternalProject_Add(pykdl_ep
          DEPENDS          orocos_kdl_ep
          SOURCE_DIR       "${_MJKDL_OROCOS_SRC}"
          SOURCE_SUBDIR    python_orocos_kdl
          DOWNLOAD_COMMAND ""
          CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX=${_MJKDL_PYKDL_INSTALL}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
            -DCMAKE_PREFIX_PATH=${_MJKDL_OROCOS_PREFIX}
            -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
            -Dpybind11_DIR=${pybind11_DIR}
            -DPYTHON_SITE_PACKAGES_INSTALL_DIR=.
            "-DCMAKE_INSTALL_RPATH=\$ORIGIN/mjkdl"
            # DT_RPATH, as for the wrapper: LD_LIBRARY_PATH must not swap in a system KDL.
            -DCMAKE_MODULE_LINKER_FLAGS=-Wl,--disable-new-dtags
            -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--disable-new-dtags
          UPDATE_DISCONNECTED TRUE
        )
        # Bundle PyKDL (top-level module) and the Orocos KDL shared library from the
        # ExternalProject install trees. FILES_MATCHING globs at install time, after
        # both ExternalProject builds have run.
        install(DIRECTORY "${_MJKDL_PYKDL_INSTALL}/"
          DESTINATION .
          FILES_MATCHING PATTERN "PyKDL*.so"
        )
        install(FILES licenses/LGPL-2.1.txt licenses/KDL_SOURCE.txt
          DESTINATION .
        )
        install(DIRECTORY "${_MJKDL_OROCOS_PREFIX}/lib/"
          DESTINATION mjkdl
          FILES_MATCHING PATTERN "liborocos-kdl.so*"
        )
        install(FILES licenses/LGPL-2.1.txt licenses/KDL_SOURCE.txt
          DESTINATION mjkdl
        )
    endif()

    pybind11_add_module(_mjkdl
      python/src/bindings.cpp
    )
    target_link_libraries(_mjkdl PRIVATE mjkdl)
    if(TARGET orocos_kdl_ep)
        # bindings.cpp includes KDL headers produced by the ExternalProject.
        add_dependencies(_mjkdl orocos_kdl_ep)
    endif()
    target_compile_features(_mjkdl PRIVATE cxx_std_20)
    target_compile_options(_mjkdl PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_definitions(_mjkdl PRIVATE
      MJKDL_VERSION="${PROJECT_VERSION}"
    )
    # The extension calls MuJoCo directly, so libmujoco is a direct dependency
    # of this module too. DT_RUNPATH is not transitive, so it needs its own
    # entry for the sibling mujoco pip package alongside $ORIGIN.
    set_target_properties(_mjkdl PROPERTIES
      BUILD_RPATH "$<TARGET_FILE_DIR:mjkdl>"
      INSTALL_RPATH "$ORIGIN;$ORIGIN/../mujoco"
    )
    install(TARGETS _mjkdl
      LIBRARY DESTINATION mjkdl
      RUNTIME DESTINATION mjkdl
    )
    install(TARGETS mjkdl
      LIBRARY DESTINATION mjkdl
      RUNTIME DESTINATION mjkdl
    )
    install(FILES
      NOTICE
      THIRD_PARTY_LICENSES.md
      licenses/Apache-2.0.txt
      licenses/Robotiq-2F85-BSD-2-Clause.txt
      licenses/Kinova-Gen3-BSD-3-Clause.txt
      DESTINATION mjkdl
    )
    # Installed, not mapped in by pyproject: a wheel mapping above this package leaks sys.path.
    install(DIRECTORY assets/
      DESTINATION mjkdl/assets
    )    # libmujoco is intentionally not bundled: the wheel depends on the matching
    # `mujoco` pip package and links its shared library at runtime via the
    # $ORIGIN/../mujoco rpath set above. This avoids shipping a second copy.
    if(EXISTS "${MJKDL_MUJOCO_DIR}/bin/mujoco_plugin")
      install(DIRECTORY "${MJKDL_MUJOCO_DIR}/bin/mujoco_plugin"
        DESTINATION mjkdl
        FILES_MATCHING PATTERN "*.so"
      )
    endif()
endif()

# Examples (GUI by default; pass --headless to skip the window)
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(src/examples)
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

    set(MJKDL_KDL_DOXYGEN_INPUT_DIRS)
    set(_mjkdl_kdl_tag_depends)
    if(_MJKDL_BUILD_OROCOS_KDL)
        # The fork's headers exist only once orocos_kdl_ep has installed them.
        set(MJKDL_KDL_DOXYGEN_INPUT_DIRS "${_MJKDL_OROCOS_PREFIX}/include/kdl")
        set(_mjkdl_kdl_tag_depends orocos_kdl_ep)
    else()
        foreach(_kdl_inc IN LISTS orocos_kdl_INCLUDE_DIRS)
            if(EXISTS "${_kdl_inc}/kdl/chain.hpp")
                list(APPEND MJKDL_KDL_DOXYGEN_INPUT_DIRS "${_kdl_inc}/kdl")
            endif()
        endforeach()
    endif()

    set(MJKDL_DOCS_DEPENDS)
    set(MJKDL_DOXYGEN_TAGFILES)
    if(MJKDL_KDL_DOXYGEN_INPUT_DIRS)
        list(REMOVE_DUPLICATES MJKDL_KDL_DOXYGEN_INPUT_DIRS)
        string(REPLACE ";" " " MJKDL_KDL_DOXYGEN_INPUTS "${MJKDL_KDL_DOXYGEN_INPUT_DIRS}")
        set(MJKDL_KDL_TAG_FILE "${CMAKE_CURRENT_BINARY_DIR}/docs/kdl.tag")
        configure_file(
          ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/Doxyfile.kdl.in
          ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.kdl
          @ONLY
        )
        add_custom_command(
          OUTPUT ${MJKDL_KDL_TAG_FILE}
          # Doxygen creates only the last level of HTML_OUTPUT (html/kdl).
          COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/docs/html
          COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.kdl
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.kdl ${_mjkdl_kdl_tag_depends}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          COMMENT "Generating Orocos KDL Doxygen tag file..."
          VERBATIM
        )
        list(APPEND MJKDL_DOCS_DEPENDS ${MJKDL_KDL_TAG_FILE})
        set(MJKDL_DOXYGEN_TAGFILES "${MJKDL_KDL_TAG_FILE}=kdl/")
        message(STATUS "Doxygen KDL tag input: ${MJKDL_KDL_DOXYGEN_INPUT_DIRS}")
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
    DEPENDS ${MJKDL_DOCS_DEPENDS}
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

set(_MJKDL_INSTALL_CPP_PACKAGE_DEFAULT ON)
if(DEFINED SKBUILD)
  set(_MJKDL_INSTALL_CPP_PACKAGE_DEFAULT OFF)
endif()
option(
  MJKDL_INSTALL_CPP_PACKAGE
  "Install the C++ library, headers, and CMake package config"
  ${_MJKDL_INSTALL_CPP_PACKAGE_DEFAULT}
)
# Make the install self-contained: also install the Orocos KDL fork we built (its
# library, headers, and CMake package config) into the same prefix, so the
# installed wrapper loads KDL via $ORIGIN and sibling packages in the prefix can
# find_package(orocos_kdl). Disable to keep KDL out of a shared prefix (e.g.
# /usr/local) where it could shadow a distro KDL. Has no effect when consuming an
# external KDL via MJKDL_OROCOS_KDL_INSTALL_DIR (that prefix is already shared).
option(
  MJKDL_INSTALL_BUNDLED_KDL
  "Install the built Orocos KDL fork into the prefix so the install is self-contained"
  ON
)

if(MJKDL_INSTALL_CPP_PACKAGE)
    install(TARGETS mjkdl
      EXPORT mjkdlTargets
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    if(TARGET mjkdl_camera_ros)
        install(TARGETS mjkdl_camera_ros
          EXPORT mjkdlTargets
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    endif()
    # The C++ install ships no assets, so only the licenses of what it compiles in.
    install(FILES
      LICENSE
      NOTICE
      THIRD_PARTY_LICENSES.md
      licenses/Apache-2.0.txt
      DESTINATION ${CMAKE_INSTALL_DOCDIR}
    )
    # Consumers read the bundled models from the user cache, not the prefix.
    if(_MJKDL_CACHE_ROOT)
        install(DIRECTORY assets/ DESTINATION "${_MJKDL_CACHE_ROOT}/assets")
        install(FILES
          licenses/Robotiq-2F85-BSD-2-Clause.txt
          licenses/Kinova-Gen3-BSD-3-Clause.txt
          DESTINATION "${_MJKDL_CACHE_ROOT}/assets"
        )
    endif()
    install(FILES include/mjkdl/mjkdl.hpp
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mjkdl
    )
    if(TARGET mjkdl_camera_ros)
        install(FILES include/mjkdl/camera_ros.hpp
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mjkdl
        )
    endif()

    # libmujoco is not bundled: the install loads it from where it was found. KDL is either
    # bundled ($ORIGIN, below) or loaded from its own prefix.
    set_property(TARGET mjkdl APPEND PROPERTY INSTALL_RPATH "${MUJOCO_LIB_DIR}")
    set(_mjkdl_kdl_dir "")
    if(NOT _MJKDL_BUILD_OROCOS_KDL)
        get_target_property(_mjkdl_kdl_type orocos-kdl TYPE)
        if(_mjkdl_kdl_type MATCHES "SHARED_LIBRARY|UNKNOWN_LIBRARY")
            get_target_property(_mjkdl_kdl_location orocos-kdl LOCATION)
            get_filename_component(_mjkdl_kdl_dir "${_mjkdl_kdl_location}" DIRECTORY)
            set_property(TARGET mjkdl APPEND PROPERTY INSTALL_RPATH "${_mjkdl_kdl_dir}")
        endif()
    elseif(NOT MJKDL_INSTALL_BUNDLED_KDL)
        message(WARNING
          "MJKDL_INSTALL_BUNDLED_KDL=OFF: the installed library has no path to the KDL fork; "
          "the loader then takes the first liborocos-kdl it finds, possibly a system one.")
    endif()
    if(TARGET mjkdl_camera_ros)
        set_property(TARGET mjkdl_camera_ros APPEND PROPERTY
          INSTALL_RPATH "$ORIGIN;${MUJOCO_LIB_DIR}")
        if(_mjkdl_kdl_dir)
            set_property(TARGET mjkdl_camera_ros APPEND PROPERTY
              INSTALL_RPATH "${_mjkdl_kdl_dir}")
        endif()
    endif()
    # Found next to the library at runtime, so a moved install keeps its plugins.
    if(EXISTS "${MJKDL_MUJOCO_DIR}/bin/mujoco_plugin")
        install(DIRECTORY "${MJKDL_MUJOCO_DIR}/bin/mujoco_plugin"
          DESTINATION ${CMAKE_INSTALL_LIBDIR}
          FILES_MATCHING PATTERN "*.so"
        )
    endif()

    # Self-contained install: bundle the fork we built (lib + headers + its CMake
    # package config) into this prefix and rpath the wrapper to $ORIGIN so it loads
    # the co-installed liborocos-kdl. Skipped when consuming an external KDL, which
    # is already shared from its own prefix.
    set(_MJKDL_KDL_IN_PREFIX OFF)
    if(MJKDL_INSTALL_BUNDLED_KDL AND _MJKDL_BUILD_OROCOS_KDL)
        set(_MJKDL_KDL_IN_PREFIX ON)
        install(DIRECTORY "${_MJKDL_OROCOS_PREFIX}/"
          DESTINATION .
          USE_SOURCE_PERMISSIONS
        )
        install(FILES licenses/LGPL-2.1.txt licenses/KDL_SOURCE.txt
          DESTINATION ${CMAKE_INSTALL_DOCDIR}
        )
        set_property(TARGET mjkdl APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
    endif()

    # KDL flags come from the fork's orocos-kdl.pc; an external KDL also needs its rpath here.
    file(RELATIVE_PATH _MJKDL_PC_PREFIX
      "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig" "${CMAKE_INSTALL_PREFIX}")
    string(REGEX REPLACE "/$" "" _MJKDL_PC_PREFIX "${_MJKDL_PC_PREFIX}")
    set(_MJKDL_PC_KDL_RPATH "")
    if(_mjkdl_kdl_dir)
        set(_MJKDL_PC_KDL_RPATH "-Wl,-rpath,${_mjkdl_kdl_dir}")
    endif()
    configure_file(cmake/mjkdl.pc.in
      ${CMAKE_CURRENT_BINARY_DIR}/mjkdl.pc @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mjkdl.pc
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    )

    install(EXPORT mjkdlTargets
      FILE mjkdlTargets.cmake
      NAMESPACE mjkdl::
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mjkdl
    )

    configure_package_config_file(
      cmake/mjkdlConfig.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/mjkdlConfig.cmake
      INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mjkdl
    )

    write_basic_package_version_file(
      ${CMAKE_CURRENT_BINARY_DIR}/mjkdlConfigVersion.cmake
      VERSION ${PROJECT_VERSION}
      COMPATIBILITY SameMinorVersion
    )

    install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/mjkdlConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/mjkdlConfigVersion.cmake
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mjkdl
    )
endif()
