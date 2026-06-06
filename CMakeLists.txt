cmake_minimum_required(VERSION 3.16)
project(mj_kdl_wrapper VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(cmake/MuJoCoVersion.cmake)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(_MJ_KDL_FETCH_MUJOCO_DEFAULT OFF)
if(DEFINED SKBUILD)
  set(_MJ_KDL_FETCH_MUJOCO_DEFAULT ON)
endif()
option(
  MJ_KDL_FETCH_MUJOCO
  "Download the supported MuJoCo release if MUJOCO_ROOT is not available"
  ${_MJ_KDL_FETCH_MUJOCO_DEFAULT}
)
set(MJ_KDL_MUJOCO_URL "" CACHE STRING "Override MuJoCo release archive URL")

set(_MJ_KDL_FETCH_OROCOS_KDL_DEFAULT OFF)
if(DEFINED SKBUILD)
  set(_MJ_KDL_FETCH_OROCOS_KDL_DEFAULT ON)
endif()
option(
  MJ_KDL_FETCH_OROCOS_KDL
  "Build Orocos KDL and PyKDL from the pinned secorolab fork"
  ${_MJ_KDL_FETCH_OROCOS_KDL_DEFAULT}
)
set(
  MJ_KDL_OROCOS_KDL_GIT_REPOSITORY
  "https://github.com/secorolab/orocos_kinematics_dynamics.git"
  CACHE STRING
  "Orocos KDL fork used for Python wheels"
)
set(
  MJ_KDL_OROCOS_KDL_GIT_TAG
  "feature/achd_fixed_joint"
  CACHE STRING
  "Orocos KDL fork branch/tag used for Python wheels"
)

# MuJoCo - prefer /opt/mujoco-<version>, override with -DMUJOCO_ROOT=...
set(MUJOCO_ROOT "${MJ_KDL_MUJOCO_DEFAULT_ROOT}" CACHE PATH "MuJoCo install root")
if(MJ_KDL_FETCH_MUJOCO AND NOT EXISTS "${MUJOCO_ROOT}/include/mujoco/mujoco.h")
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR
          "Automatic MuJoCo download is currently supported on Linux only. "
          "Set MUJOCO_ROOT to an existing MuJoCo ${MJ_KDL_MUJOCO_VERSION} install.")
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
        set(_mj_kdl_mujoco_platform "linux-x86_64")
    else()
        message(FATAL_ERROR
          "Automatic MuJoCo download does not support ${CMAKE_SYSTEM_PROCESSOR}. "
          "Set MUJOCO_ROOT to an existing MuJoCo ${MJ_KDL_MUJOCO_VERSION} install.")
    endif()
    if(NOT MJ_KDL_MUJOCO_URL)
        set(
          MJ_KDL_MUJOCO_URL
          "https://github.com/google-deepmind/mujoco/releases/download/${MJ_KDL_MUJOCO_VERSION}/mujoco-${MJ_KDL_MUJOCO_VERSION}-${_mj_kdl_mujoco_platform}.tar.gz"
          CACHE STRING
          "Override MuJoCo release archive URL"
          FORCE
        )
    endif()
    include(FetchContent)
    if(POLICY CMP0135)
      cmake_policy(SET CMP0135 NEW)
    endif()
    if(POLICY CMP0169)
      cmake_policy(SET CMP0169 OLD)
    endif()
    message(STATUS "MuJoCo : fetching ${MJ_KDL_MUJOCO_URL}")
    FetchContent_Declare(
      mujoco_release
      URL "${MJ_KDL_MUJOCO_URL}"
    )
    FetchContent_Populate(mujoco_release)
    set(MUJOCO_ROOT "${mujoco_release_SOURCE_DIR}" CACHE PATH "MuJoCo install root" FORCE)
endif()
find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
          PATHS "${MUJOCO_ROOT}/include" /usr/local/include /usr/include
          NO_DEFAULT_PATH REQUIRED)
file(STRINGS "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h" MUJOCO_VERSION_HEADER_LINE
     REGEX "^#define mjVERSION_HEADER ")
string(REGEX REPLACE ".*mjVERSION_HEADER +([0-9]+).*" "\\1" MUJOCO_VERSION_HEADER
       "${MUJOCO_VERSION_HEADER_LINE}")
if(NOT MUJOCO_VERSION_HEADER STREQUAL "${MJ_KDL_MUJOCO_VERSION_HEADER}")
    message(FATAL_ERROR
      "mj_kdl_wrapper requires MuJoCo ${MJ_KDL_MUJOCO_VERSION} "
      "(mjVERSION_HEADER ${MJ_KDL_MUJOCO_VERSION_HEADER}), "
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

if(MJ_KDL_FETCH_OROCOS_KDL)
    include(FetchContent)
    if(POLICY CMP0135)
      cmake_policy(SET CMP0135 NEW)
    endif()
    if(POLICY CMP0169)
      cmake_policy(SET CMP0169 OLD)
    endif()
    message(
      STATUS
      "Orocos KDL: fetching ${MJ_KDL_OROCOS_KDL_GIT_REPOSITORY} (${MJ_KDL_OROCOS_KDL_GIT_TAG})"
    )
    FetchContent_Declare(
      orocos_kinematics_dynamics
      GIT_REPOSITORY "${MJ_KDL_OROCOS_KDL_GIT_REPOSITORY}"
      GIT_TAG "${MJ_KDL_OROCOS_KDL_GIT_TAG}"
      GIT_SHALLOW TRUE
    )
    FetchContent_Populate(orocos_kinematics_dynamics)

    set(_MJ_KDL_OROCOS_PREFIX "${CMAKE_BINARY_DIR}/orocos_kdl_install")
    set(_MJ_KDL_OROCOS_BUILD "${CMAKE_BINARY_DIR}/orocos_kdl_build")
    execute_process(
      COMMAND
        "${CMAKE_COMMAND}"
        -S "${orocos_kinematics_dynamics_SOURCE_DIR}/orocos_kdl"
        -B "${_MJ_KDL_OROCOS_BUILD}"
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${_MJ_KDL_OROCOS_PREFIX}
        -DENABLE_TESTS=OFF
        -DENABLE_EXAMPLES=OFF
      COMMAND_ERROR_IS_FATAL ANY
    )
    execute_process(
      COMMAND
        "${CMAKE_COMMAND}"
        --build "${_MJ_KDL_OROCOS_BUILD}"
        --target install
        --parallel
      COMMAND_ERROR_IS_FATAL ANY
    )
    list(PREPEND CMAKE_PREFIX_PATH "${_MJ_KDL_OROCOS_PREFIX}")
endif()

if(MJ_KDL_FETCH_OROCOS_KDL)
    # When fetching, the pinned secorolab fork is the only allowed KDL. Restrict
    # the search to its install prefix so a distro liborocos-kdl is never used.
    find_package(orocos_kdl REQUIRED CONFIG
      PATHS "${_MJ_KDL_OROCOS_PREFIX}"
      NO_DEFAULT_PATH
    )
else()
    find_package(orocos_kdl REQUIRED CONFIG
      NO_CMAKE_PACKAGE_REGISTRY
    )
endif()

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

# Optional viewer UI sections (hidden by default)
option(SHOW_EQUALITY_PANEL "Show the Equality UI section in the viewer" OFF)
option(SHOW_GROUP_PANEL    "Show the Group-enable UI section in the viewer" OFF)

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
if(SHOW_EQUALITY_PANEL)
  target_compile_definitions(mujoco_simulate_ui PRIVATE MJ_KDL_SHOW_EQUALITY)
endif()
if(SHOW_GROUP_PANEL)
  target_compile_definitions(mujoco_simulate_ui PRIVATE MJ_KDL_SHOW_GROUP)
endif()
set_target_properties(mujoco_simulate_ui PROPERTIES
  CXX_STANDARD 20 POSITION_INDEPENDENT_CODE ON)

# Library
add_library(mj_kdl_wrapper SHARED
  src/mj_kdl_wrapper.cpp
)

target_include_directories(mj_kdl_wrapper PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
if(IS_ABSOLUTE "${MUJOCO_INCLUDE_DIR}" AND "${MUJOCO_INCLUDE_DIR}" MATCHES "^${CMAKE_BINARY_DIR}")
    target_include_directories(mj_kdl_wrapper PUBLIC $<BUILD_INTERFACE:${MUJOCO_INCLUDE_DIR}>)
else()
    target_include_directories(mj_kdl_wrapper PUBLIC "${MUJOCO_INCLUDE_DIR}")
endif()
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

option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
if(BUILD_PYTHON_BINDINGS)
    set_target_properties(mj_kdl_wrapper PROPERTIES
      BUILD_RPATH "${MUJOCO_LIB_DIR}"
      INSTALL_RPATH "$ORIGIN"
    )
    target_compile_definitions(mj_kdl_wrapper PRIVATE MJ_KDL_RELOCATABLE_PLUGINS)
    target_link_libraries(mj_kdl_wrapper PRIVATE ${CMAKE_DL_LIBS})

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

    if(MJ_KDL_FETCH_OROCOS_KDL)
        set(_MJ_KDL_PYKDL_BUILD "${CMAKE_BINARY_DIR}/python_orocos_kdl_build")
        set(_MJ_KDL_PYKDL_INSTALL "${CMAKE_BINARY_DIR}/python_orocos_kdl_install")
        execute_process(
          COMMAND
            "${CMAKE_COMMAND}"
            -S "${orocos_kinematics_dynamics_SOURCE_DIR}/python_orocos_kdl"
            -B "${_MJ_KDL_PYKDL_BUILD}"
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX=${_MJ_KDL_PYKDL_INSTALL}
            -DCMAKE_PREFIX_PATH=${_MJ_KDL_OROCOS_PREFIX}
            -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
            -Dpybind11_DIR=${pybind11_DIR}
            -DPYTHON_SITE_PACKAGES_INSTALL_DIR=.
            "-DCMAKE_INSTALL_RPATH=\$ORIGIN/mj_kdl_wrapper"
          COMMAND_ERROR_IS_FATAL ANY
        )
        execute_process(
          COMMAND
            "${CMAKE_COMMAND}"
            --build "${_MJ_KDL_PYKDL_BUILD}"
            --target install
            --parallel
          COMMAND_ERROR_IS_FATAL ANY
        )
        file(GLOB _MJ_KDL_PYKDL_MODULES "${_MJ_KDL_PYKDL_INSTALL}/PyKDL*.so")
        if(NOT _MJ_KDL_PYKDL_MODULES)
            message(FATAL_ERROR "PyKDL build did not install a PyKDL extension")
        endif()
        install(FILES ${_MJ_KDL_PYKDL_MODULES}
          DESTINATION .
        )
        install(DIRECTORY "${_MJ_KDL_OROCOS_PREFIX}/lib/"
          DESTINATION mj_kdl_wrapper
          FILES_MATCHING PATTERN "liborocos-kdl.so*"
        )
    endif()

    pybind11_add_module(_mj_kdl_wrapper
      src/python/bindings.cpp
    )
    target_link_libraries(_mj_kdl_wrapper PRIVATE mj_kdl_wrapper)
    target_compile_features(_mj_kdl_wrapper PRIVATE cxx_std_20)
    target_compile_options(_mj_kdl_wrapper PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(_mj_kdl_wrapper PROPERTIES
      BUILD_RPATH "$<TARGET_FILE_DIR:mj_kdl_wrapper>"
      INSTALL_RPATH "$ORIGIN"
    )
    install(TARGETS _mj_kdl_wrapper
      LIBRARY DESTINATION mj_kdl_wrapper
      RUNTIME DESTINATION mj_kdl_wrapper
    )
    install(TARGETS mj_kdl_wrapper
      LIBRARY DESTINATION mj_kdl_wrapper
      RUNTIME DESTINATION mj_kdl_wrapper
    )
    get_filename_component(MUJOCO_LIBRARY_REALPATH "${MUJOCO_LIBRARY}" REALPATH)
    install(FILES "${MUJOCO_LIBRARY_REALPATH}"
      DESTINATION mj_kdl_wrapper
    )
    if(EXISTS "${MUJOCO_ROOT}/bin/mujoco_plugin")
      install(DIRECTORY "${MUJOCO_ROOT}/bin/mujoco_plugin"
        DESTINATION mj_kdl_wrapper
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

# Documentation (Doxygen + local CSS)
# Build separately:  cmake --build build --target docs
# Requires:  sudo apt install doxygen
option(BUILD_DOCS "Generate HTML documentation with Doxygen" OFF)
if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)

    set(MJ_KDL_KDL_DOXYGEN_INPUT_DIRS)
    foreach(_kdl_inc IN LISTS orocos_kdl_INCLUDE_DIRS)
        if(EXISTS "${_kdl_inc}/kdl")
            list(APPEND MJ_KDL_KDL_DOXYGEN_INPUT_DIRS "${_kdl_inc}/kdl")
        elseif(EXISTS "${_kdl_inc}/chain.hpp" OR EXISTS "${_kdl_inc}/frames.hpp")
            list(APPEND MJ_KDL_KDL_DOXYGEN_INPUT_DIRS "${_kdl_inc}")
        endif()
    endforeach()
    if(NOT MJ_KDL_KDL_DOXYGEN_INPUT_DIRS)
        find_path(MJ_KDL_KDL_INCLUDE_ROOT kdl/chain.hpp
          PATHS ${orocos_kdl_INCLUDE_DIRS} /usr/include /usr/local/include
        )
        if(MJ_KDL_KDL_INCLUDE_ROOT AND EXISTS "${MJ_KDL_KDL_INCLUDE_ROOT}/kdl")
            list(APPEND MJ_KDL_KDL_DOXYGEN_INPUT_DIRS "${MJ_KDL_KDL_INCLUDE_ROOT}/kdl")
        endif()
    endif()

    set(MJ_KDL_DOCS_DEPENDS)
    set(MJ_KDL_DOXYGEN_TAGFILES)
    if(MJ_KDL_KDL_DOXYGEN_INPUT_DIRS)
        list(REMOVE_DUPLICATES MJ_KDL_KDL_DOXYGEN_INPUT_DIRS)
        string(REPLACE ";" " " MJ_KDL_KDL_DOXYGEN_INPUTS "${MJ_KDL_KDL_DOXYGEN_INPUT_DIRS}")
        set(MJ_KDL_KDL_TAG_FILE "${CMAKE_CURRENT_BINARY_DIR}/docs/kdl.tag")
        configure_file(
          ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.kdl.in
          ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.kdl
          @ONLY
        )
        add_custom_command(
          OUTPUT ${MJ_KDL_KDL_TAG_FILE}
          COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/docs
          COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.kdl
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.kdl
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          COMMENT "Generating Orocos KDL Doxygen tag file..."
          VERBATIM
        )
        list(APPEND MJ_KDL_DOCS_DEPENDS ${MJ_KDL_KDL_TAG_FILE})
        set(MJ_KDL_DOXYGEN_TAGFILES "${MJ_KDL_KDL_TAG_FILE}=kdl/")
        message(STATUS "Doxygen KDL tag input: ${MJ_KDL_KDL_DOXYGEN_INPUT_DIRS}")
    else()
        message(WARNING "Doxygen: KDL headers were not found; external KDL symbols will not be resolved")
    endif()

    configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
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

if(MJ_KDL_INSTALL_CPP_PACKAGE)
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
endif()
