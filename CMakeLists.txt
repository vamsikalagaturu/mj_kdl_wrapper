cmake_minimum_required(VERSION 3.16)
project(mj_kdl_wrapper VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# MuJoCo — prefer /opt/mujoco-3.8.0, override with -DMUJOCO_ROOT=...
set(MUJOCO_ROOT "/opt/mujoco-3.8.0" CACHE PATH "MuJoCo install root")
find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
          PATHS "${MUJOCO_ROOT}/include" /usr/local/include /usr/include
          NO_DEFAULT_PATH REQUIRED)
file(STRINGS "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h" MUJOCO_VERSION_HEADER_LINE
     REGEX "^#define mjVERSION_HEADER ")
string(REGEX REPLACE ".*mjVERSION_HEADER +([0-9]+).*" "\\1" MUJOCO_VERSION_HEADER
       "${MUJOCO_VERSION_HEADER_LINE}")
if(NOT MUJOCO_VERSION_HEADER STREQUAL "3008000")
    message(FATAL_ERROR
      "mj_kdl_wrapper requires MuJoCo 3.8.0 (mjVERSION_HEADER 3008000), "
      "but found ${MUJOCO_VERSION_HEADER} at ${MUJOCO_INCLUDE_DIR}")
endif()
find_library(MUJOCO_LIBRARY mujoco
             PATHS "${MUJOCO_ROOT}/lib" /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
             NO_DEFAULT_PATH REQUIRED)
get_filename_component(MUJOCO_LIB_DIR "${MUJOCO_LIBRARY}" DIRECTORY)
list(APPEND CMAKE_BUILD_RPATH "${MUJOCO_LIB_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
message(STATUS "MuJoCo : ${MUJOCO_LIBRARY}")

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

find_package(orocos_kdl REQUIRED CONFIG
  NO_CMAKE_PACKAGE_REGISTRY
)

set(_KDL_PARSER_INSTALL "${CMAKE_SOURCE_DIR}/../../install/kdl_parser")
get_filename_component(_KDL_PARSER_INSTALL "${_KDL_PARSER_INSTALL}" ABSOLUTE)
find_library(KDL_PARSER_LIB kdl_parser
  PATHS "${_KDL_PARSER_INSTALL}/lib"
  NO_DEFAULT_PATH)
if(KDL_PARSER_LIB)
  add_library(kdl_parser::kdl_parser SHARED IMPORTED)
  set_target_properties(kdl_parser::kdl_parser PROPERTIES
    IMPORTED_LOCATION "${KDL_PARSER_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${_KDL_PARSER_INSTALL}/include"
  )
  set(kdl_parser_FOUND TRUE)
  set(_KDL_PARSER_LIB_DIR "${_KDL_PARSER_INSTALL}/lib")
  message(STATUS "kdl_parser (local): ${KDL_PARSER_LIB}")
endif()

if(NOT orocos_kdl_FOUND)
    message(FATAL_ERROR
    "orocos_kdl not found. Set either:\n"
    "  -Dorocos_kdl_DIR=/path/to/orocos_kdlConfig.cmake-dir\n"
    "or\n"
    "  -DCMAKE_PREFIX_PATH=/path/to/install/prefix")
endif()

message(STATUS "KDL    : ${orocos_kdl_LIBRARIES}")
message(STATUS "KDL inc: ${orocos_kdl_INCLUDE_DIRS}")
message(STATUS "KDL cfg: ${orocos_kdl_DIR}")


# lodepng stub + ffmpeg-backed encoder -- satisfies the lodepng::encode() call in
# simulate.cc without pulling in the full lodepng library.
set(LODEPNG_STUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/lodepng_stub)

set(MJ_SIM_DIR "${MUJOCO_ROOT}/simulate")
set(MJ_SIM_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/simulate_ui")
set(SIMULATE_UI_SOURCES
  ${MJ_SIM_LOCAL_DIR}/simulate.cc
  ${MJ_SIM_DIR}/glfw_adapter.cc
  ${MJ_SIM_DIR}/glfw_dispatch.cc
  ${MJ_SIM_DIR}/platform_ui_adapter.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/lodepng_ffmpeg.cpp
)
message(STATUS "lodepng: using ffmpeg-backed stub - UI screenshots enabled")

add_library(mujoco_simulate_ui STATIC ${SIMULATE_UI_SOURCES})
target_include_directories(mujoco_simulate_ui PUBLIC
  ${MJ_SIM_LOCAL_DIR}
  ${MJ_SIM_DIR}
  ${MUJOCO_INCLUDE_DIR}
)
target_link_libraries(mujoco_simulate_ui PUBLIC
  ${MUJOCO_LIBRARY}
  glfw
  OpenGL::GL
)
target_include_directories(mujoco_simulate_ui PRIVATE
  ${LODEPNG_STUB_DIR}
)
target_compile_options(mujoco_simulate_ui PRIVATE -w)
set_target_properties(mujoco_simulate_ui PROPERTIES
  CXX_STANDARD 20 POSITION_INDEPENDENT_CODE ON)

# Library
add_library(mj_kdl_wrapper SHARED
  src/mj_kdl_wrapper.cpp
)

target_include_directories(mj_kdl_wrapper PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${MUJOCO_INCLUDE_DIR}
)
# orocos_kdl include dirs may live inside the source tree (e.g. a local install/).
# Wrap in BUILD_INTERFACE only when the path is inside CMAKE_SOURCE_DIR to satisfy
# CMake's install-interface path check; add other paths (system, workspace) directly.
foreach(_kdl_inc IN LISTS orocos_kdl_INCLUDE_DIRS)
    if(IS_ABSOLUTE "${_kdl_inc}" AND "${_kdl_inc}" MATCHES "^${CMAKE_SOURCE_DIR}")
        target_include_directories(mj_kdl_wrapper PUBLIC $<BUILD_INTERFACE:${_kdl_inc}>)
    else()
        target_include_directories(mj_kdl_wrapper PUBLIC "${_kdl_inc}")
    endif()
endforeach()
target_include_directories(mj_kdl_wrapper PRIVATE
  ${MJ_SIM_LOCAL_DIR}
  ${MJ_SIM_DIR}
  ${LODEPNG_STUB_DIR}
)

target_link_libraries(mj_kdl_wrapper PUBLIC
  ${MUJOCO_LIBRARY}
  glfw
  OpenGL::GL
  ${orocos_kdl_LIBRARIES}
)
target_link_libraries(mj_kdl_wrapper PRIVATE
  mujoco_simulate_ui
)

target_compile_options(mj_kdl_wrapper PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(mj_kdl_wrapper PRIVATE
  MUJOCO_PLUGIN_DIR="${MUJOCO_ROOT}/bin/mujoco_plugin"
)
if(BUILD_RECORDER AND EGL_LIBRARY)
    target_link_libraries(mj_kdl_wrapper PUBLIC ${EGL_LIBRARY})
    target_compile_definitions(mj_kdl_wrapper PRIVATE MJ_KDL_HAS_EGL)
endif()

# Examples (GUI by default; pass --headless to skip the window)
add_subdirectory(src/examples)

# Tests
option(BUILD_TESTS "Build and register tests with CTest" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# MuJoCo Menagerie (robot model zoo - needed for test_kinova_gen3_menagerie)
option(FETCH_MENAGERIE "Download MuJoCo Menagerie into third_party/menagerie" OFF)
if(FETCH_MENAGERIE)
    set(MENAGERIE_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/menagerie)
    if(EXISTS ${MENAGERIE_SOURCE_DIR}/README.md)
        message(STATUS "Menagerie: already present at ${MENAGERIE_SOURCE_DIR}")
    else()
        include(FetchContent)
        message(STATUS "Menagerie: fetching into ${MENAGERIE_SOURCE_DIR}")
        FetchContent_Declare(
      mujoco_menagerie
      GIT_REPOSITORY https://github.com/google-deepmind/mujoco_menagerie.git
      GIT_TAG        main
      GIT_SHALLOW    TRUE
      SOURCE_DIR     ${MENAGERIE_SOURCE_DIR}
    )
        FetchContent_MakeAvailable(mujoco_menagerie)
        message(STATUS "Menagerie: ready at ${MENAGERIE_SOURCE_DIR}")
    endif()
endif()

# Documentation (Doxygen + Doxygen Awesome CSS)
# Build separately:  cmake --build build --target docs
# Requires:  sudo apt install doxygen
option(BUILD_DOCS "Generate HTML documentation with Doxygen and Awesome CSS" OFF)
if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)

    # Doxygen Awesome CSS - modern responsive theme for Doxygen
    set(DOXYGEN_AWESOME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doxygen-awesome-css)
    if(NOT EXISTS ${DOXYGEN_AWESOME_DIR}/doxygen-awesome.css)
        message(STATUS "doxygen-awesome: fetching into ${DOXYGEN_AWESOME_DIR}")
        include(FetchContent)
        FetchContent_Declare(
      doxygen_awesome
      GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
      GIT_TAG        v2.4.2
      GIT_SHALLOW    TRUE
      SOURCE_DIR     ${DOXYGEN_AWESOME_DIR}
    )
        FetchContent_MakeAvailable(doxygen_awesome)
    endif()

    configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    @ONLY
  )

    add_custom_target(docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${DOXYGEN_AWESOME_DIR}/doxygen-awesome-darkmode-toggle.js
      ${CMAKE_CURRENT_SOURCE_DIR}/docs/html/doxygen-awesome-darkmode-toggle.js
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${DOXYGEN_AWESOME_DIR}/doxygen-awesome-paragraph-link.js
      ${CMAKE_CURRENT_SOURCE_DIR}/docs/html/doxygen-awesome-paragraph-link.js
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${DOXYGEN_AWESOME_DIR}/doxygen-awesome-interactive-toc.js
      ${CMAKE_CURRENT_SOURCE_DIR}/docs/html/doxygen-awesome-interactive-toc.js
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${DOXYGEN_AWESOME_DIR}/doxygen-awesome.css
      ${CMAKE_CURRENT_SOURCE_DIR}/docs/html/doxygen-awesome.css
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${DOXYGEN_AWESOME_DIR}/doxygen-awesome-sidebar-only.css
      ${CMAKE_CURRENT_SOURCE_DIR}/docs/html/doxygen-awesome-sidebar-only.css
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${DOXYGEN_AWESOME_DIR}/doxygen-awesome-sidebar-only-darkmode-toggle.css
      ${CMAKE_CURRENT_SOURCE_DIR}/docs/html/doxygen-awesome-sidebar-only-darkmode-toggle.css
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${DOXYGEN_AWESOME_DIR}/doxygen-custom/theme-robot.css
      ${CMAKE_CURRENT_SOURCE_DIR}/docs/html/theme-robot.css
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${DOXYGEN_AWESOME_DIR}/doxygen-custom/custom.css
      ${CMAKE_CURRENT_SOURCE_DIR}/docs/html/custom.css
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_CURRENT_SOURCE_DIR}/docs/screenshots
      ${CMAKE_CURRENT_SOURCE_DIR}/docs/html/docs/screenshots
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen..."
    VERBATIM
  )
    message(STATUS "Doxygen: docs target enabled (run: cmake --build build --target docs)")
endif()

# Install
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS mj_kdl_wrapper
  EXPORT mj_kdl_wrapperTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

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
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/mj_kdl_wrapperConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/mj_kdl_wrapperConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mj_kdl_wrapper
)
