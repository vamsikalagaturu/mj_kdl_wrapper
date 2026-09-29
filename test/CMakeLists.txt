include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        ${MJ_KDL_GOOGLETEST_GIT_TAG})
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# Tests on Menagerie or bundled models self-skip via GTEST_SKIP() when the cache lacks them.
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
  gtest_discover_tests(${test_name} DISCOVERY_TIMEOUT 60 PROPERTIES RUN_SERIAL TRUE)
  if(TARGET mj_kdl_cache_assets)
    add_dependencies(${test_name} mj_kdl_cache_assets)
  endif()
endforeach()
foreach(test_name test_control_modes test_mjcf_load)
  target_compile_definitions(${test_name} PRIVATE
    MJ_KDL_TEST_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
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
