include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        ${MJ_KDL_GOOGLETEST_GIT_TAG})
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

set(ALL_TESTS
  test_init
  test_dual_arm
  test_table_scene
  test_mjcf_load
  test_mjcf_pos_ctrl
  test_mjcf_vel_ctrl
  test_mjcf_trq_ctrl
  test_mjcf_ft_sensor
  test_mjcf_pick
  test_scene_state
  test_control_modes
  test_regressions
)

foreach(test_name ${ALL_TESTS})
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE mj_kdl_wrapper GTest::gtest)
  target_include_directories(${test_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src/examples")
  target_compile_options(${test_name} PRIVATE -Wall -Wextra)
  # RUN_SERIAL avoids conflicts over shared temp files and MuJoCo global XML state.
  target_compile_definitions(${test_name} PRIVATE
    MJ_KDL_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets")
  gtest_discover_tests(${test_name} DISCOVERY_TIMEOUT 60 PROPERTIES RUN_SERIAL TRUE)
endforeach()
foreach(test_name test_control_modes test_mjcf_load)
  target_compile_definitions(${test_name} PRIVATE
    MJ_KDL_TEST_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
endforeach()

# The optional components' tests follow their targets.
set(_mj_kdl_component_tests)
if(TARGET mj_kdl_wrapper::render)
  list(APPEND _mj_kdl_component_tests test_render)
  set(test_render_link mj_kdl_wrapper::render)
endif()
if(TARGET mj_kdl_wrapper::viewer)
  list(APPEND _mj_kdl_component_tests test_viewer)
  set(test_viewer_link mj_kdl_wrapper::viewer)
endif()
foreach(test_name ${_mj_kdl_component_tests})
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE ${${test_name}_link} GTest::gtest)
  target_include_directories(${test_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src/examples")
  target_compile_options(${test_name} PRIVATE -Wall -Wextra)
  target_compile_definitions(${test_name} PRIVATE
    MJ_KDL_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets")
  gtest_discover_tests(${test_name} DISCOVERY_TIMEOUT 60 PROPERTIES RUN_SERIAL TRUE)
endforeach()

# The examples self-check under --headless and exit non-zero on failure.
if(BUILD_EXAMPLES)
  foreach(ex ${MJ_KDL_EXAMPLES})
    add_test(NAME ${ex}_headless COMMAND ${ex} --headless)
    set_tests_properties(${ex}_headless PROPERTIES RUN_SERIAL TRUE TIMEOUT 300)
  endforeach()
endif()

# The camera publisher only exists when the wrapper was configured with ROS, so its test
# follows the target rather than the list above.
if(TARGET mj_kdl_wrapper::camera_ros)
  add_executable(test_camera_ros test_camera_ros.cpp)
  target_link_libraries(test_camera_ros PRIVATE mj_kdl_wrapper::camera_ros GTest::gtest)
  target_compile_options(test_camera_ros PRIVATE -Wall -Wextra)
  gtest_discover_tests(test_camera_ros DISCOVERY_TIMEOUT 60 PROPERTIES RUN_SERIAL TRUE)
endif()
