include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# All tests use the MuJoCo Menagerie MJCF (kinova_gen3/gen3.xml).
# Each test self-skips via GTEST_SKIP() when Menagerie is absent.
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
)

foreach(test_name ${ALL_TESTS})
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE mj_kdl_wrapper GTest::gtest)
  target_include_directories(${test_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src/examples")
  target_compile_options(${test_name} PRIVATE -Wall -Wextra)
  # RUN_SERIAL avoids conflicts over shared temp files and MuJoCo global XML state.
  gtest_discover_tests(${test_name} DISCOVERY_TIMEOUT 60 PROPERTIES RUN_SERIAL TRUE)
endforeach()
foreach(test_name test_control_modes test_mjcf_load)
  target_compile_definitions(${test_name} PRIVATE
    MJ_KDL_TEST_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
endforeach()

# The examples self-check under --headless and exit non-zero on failure. ex_record is left out:
# it only writes a video.
if(BUILD_EXAMPLES)
  set(HEADLESS_EXAMPLES
    ex_gravity_comp
    ex_pos_ctrl
    ex_vel_ctrl
    ex_impedance
    ex_admittance_ft
    ex_admittance_ft_rnea
    ex_admittance_ft_achd
    ex_pick
    ex_table_pick_place
    ex_rnea_pick_place
    ex_achd_table_slide
    ex_achd_pick_place
    ex_achd_press
    ex_table_pour
    ex_table_scene
    ex_dual_arm
  )
  foreach(ex ${HEADLESS_EXAMPLES})
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
  list(APPEND ALL_TESTS test_camera_ros)
endif()

install(TARGETS ${ALL_TESTS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
