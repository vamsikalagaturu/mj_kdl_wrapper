cmake_minimum_required(VERSION 3.16)
project(mj_kdl_wrapper VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# MuJoCo — prefer /opt/mujoco-3.5.0, override with -DMUJOCO_ROOT=...
set(MUJOCO_ROOT "/opt/mujoco-3.5.0" CACHE PATH "MuJoCo install root")
find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
          PATHS "${MUJOCO_ROOT}/include" /usr/local/include /usr/include
          NO_DEFAULT_PATH REQUIRED)
find_library(MUJOCO_LIBRARY mujoco
             PATHS "${MUJOCO_ROOT}/lib" /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
             NO_DEFAULT_PATH REQUIRED)
get_filename_component(MUJOCO_LIB_DIR "${MUJOCO_LIBRARY}" DIRECTORY)
list(APPEND CMAKE_BUILD_RPATH "${MUJOCO_LIB_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
message(STATUS "MuJoCo : ${MUJOCO_LIBRARY}")

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

# orocos-kdl (apt: liborocos-kdl-dev)
find_package(orocos_kdl REQUIRED)
message(STATUS "KDL    : ${orocos_kdl_LIBRARIES}")

# urdfdom (apt: liburdfdom-dev)
find_package(urdfdom REQUIRED)
message(STATUS "urdfdom: ${urdfdom_LIBRARIES}")

# tinyxml2 (apt: libtinyxml2-dev)
find_package(tinyxml2 QUIET)
if(NOT tinyxml2_FOUND)
  find_path(TINYXML2_INCLUDE_DIR tinyxml2.h
            PATHS /usr/include /usr/local/include REQUIRED)
  find_library(TINYXML2_LIBRARY tinyxml2
               PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu REQUIRED)
  add_library(tinyxml2::tinyxml2 UNKNOWN IMPORTED)
  set_target_properties(tinyxml2::tinyxml2 PROPERTIES
    IMPORTED_LOCATION "${TINYXML2_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TINYXML2_INCLUDE_DIR}")
endif()

# ---------------------------------------------------------------------------
# Bundled kdl_parser (ros/kdl_parser jazzy, rcutils stripped)
# ---------------------------------------------------------------------------
add_library(kdl_parser STATIC
  third_party/kdl_parser/src/kdl_parser.cpp
)
set_target_properties(kdl_parser PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(kdl_parser PUBLIC
  third_party/kdl_parser/include
  ${orocos_kdl_INCLUDE_DIRS}
  ${urdfdom_INCLUDE_DIRS}
)
target_link_libraries(kdl_parser PUBLIC
  ${orocos_kdl_LIBRARIES}
  ${urdfdom_LIBRARIES}
)
target_compile_options(kdl_parser PRIVATE -w)  # suppress warnings in vendored code

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(mj_kdl_wrapper SHARED
  src/mj_kdl_wrapper.cpp
)

target_include_directories(mj_kdl_wrapper PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${MUJOCO_INCLUDE_DIR}
  ${orocos_kdl_INCLUDE_DIRS}
  ${urdfdom_INCLUDE_DIRS}
)

target_link_libraries(mj_kdl_wrapper PUBLIC
  ${MUJOCO_LIBRARY}
  glfw
  OpenGL::GL
  kdl_parser
  tinyxml2::tinyxml2
)

target_compile_options(mj_kdl_wrapper PRIVATE -Wall -Wextra -Wpedantic)

# ---------------------------------------------------------------------------
# MuJoCo simulate UI  (simulate.cc + GLFW adapter from the MuJoCo source tree)
# ---------------------------------------------------------------------------
set(MJ_SIM_DIR "${MUJOCO_ROOT}/simulate")
add_library(mujoco_simulate_ui STATIC
  ${MJ_SIM_DIR}/simulate.cc
  ${MJ_SIM_DIR}/glfw_adapter.cc
  ${MJ_SIM_DIR}/glfw_dispatch.cc
  ${MJ_SIM_DIR}/platform_ui_adapter.cc
)
target_include_directories(mujoco_simulate_ui PUBLIC
  ${MJ_SIM_DIR}
  ${MUJOCO_INCLUDE_DIR}
)
target_link_libraries(mujoco_simulate_ui PUBLIC
  ${MUJOCO_LIBRARY}
  glfw
  OpenGL::GL
)
target_include_directories(mujoco_simulate_ui PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lodepng   # stub for screenshot saving
)
target_compile_options(mujoco_simulate_ui PRIVATE -w)
set_target_properties(mujoco_simulate_ui PROPERTIES
  CXX_STANDARD 17 POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Simulate UI helpers  (run_simulate_ui — real-time PhysicsLoop wrapper)
# ---------------------------------------------------------------------------
add_library(mj_kdl_simulate STATIC
  src/simulate_ui.cpp
)
target_include_directories(mj_kdl_simulate PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${MJ_SIM_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lodepng
)
target_link_libraries(mj_kdl_simulate PUBLIC
  mj_kdl_wrapper
  mujoco_simulate_ui
)
target_compile_options(mj_kdl_simulate PRIVATE -w)
set_target_properties(mj_kdl_simulate PROPERTIES CXX_STANDARD 17)

# ---------------------------------------------------------------------------
# Tests  (all link mj_kdl_simulate which transitively pulls mj_kdl_wrapper)
# ---------------------------------------------------------------------------
foreach(test_name test_init test_velocity test_gravity_comp test_dual_arm test_table_scene
                  test_kinova_gen3_menagerie test_kinova_gen3_gripper
                  test_kinova_gen3_impedance test_kinova_gen3_pick)
  add_executable(${test_name} test/${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE mj_kdl_simulate)
  target_compile_options(${test_name} PRIVATE -Wall -Wextra)
endforeach()

add_executable(export_table_scene tools/export_table_scene.cpp)
target_link_libraries(export_table_scene PRIVATE mj_kdl_wrapper)
target_compile_options(export_table_scene PRIVATE -Wall -Wextra)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS mj_kdl_wrapper test_init test_velocity test_gravity_comp test_dual_arm test_table_scene
                test_kinova_gen3_menagerie test_kinova_gen3_gripper
                test_kinova_gen3_impedance test_kinova_gen3_pick
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
