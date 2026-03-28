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
# Each test self-skips via GTEST_SKIP() when third_party/menagerie is absent.
set(ALL_TESTS
  test_init
  test_dual_arm
  test_table_scene
  test_mjcf_load
  test_mjcf_pos_ctrl
  test_mjcf_vel_ctrl
  test_mjcf_trq_ctrl
  test_mjcf_pick
)

foreach(test_name ${ALL_TESTS})
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE mj_kdl_wrapper GTest::gtest)
  target_compile_options(${test_name} PRIVATE -Wall -Wextra)
  # RUN_SERIAL avoids conflicts over shared temp files and MuJoCo global XML state.
  gtest_discover_tests(${test_name} DISCOVERY_TIMEOUT 60 PROPERTIES RUN_SERIAL TRUE)
endforeach()

install(TARGETS ${ALL_TESTS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
