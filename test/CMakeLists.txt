include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

set(CORE_TESTS
  test_init
  test_velocity
  test_gravity_comp
  test_dual_arm
  test_table_scene
)

# Tests that require third_party/menagerie (MuJoCo Menagerie submodule).
# They self-skip at runtime via GTEST_SKIP() when the submodule is absent.
set(MENAGERIE_TESTS
  test_kinova_gen3_menagerie
  test_kinova_gen3_gripper
  test_kinova_gen3_impedance
  test_kinova_gen3_pick
)

foreach(test_name ${CORE_TESTS})
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE mj_kdl_wrapper GTest::gtest)
  target_compile_options(${test_name} PRIVATE -Wall -Wextra)
  # RUN_SERIAL avoids conflicts over shared temp files (e.g. _mj_kdl_r0_proc.urdf)
  # when CTest runs individual GTest cases in parallel.
  gtest_discover_tests(${test_name} PROPERTIES RUN_SERIAL TRUE)
endforeach()

foreach(test_name ${MENAGERIE_TESTS})
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE mj_kdl_wrapper GTest::gtest)
  target_compile_options(${test_name} PRIVATE -Wall -Wextra)
  # RUN_SERIAL also guards against MuJoCo global XML state conflicts in parallel.
  gtest_discover_tests(${test_name} DISCOVERY_TIMEOUT 60 PROPERTIES RUN_SERIAL TRUE)
endforeach()

install(TARGETS ${CORE_TESTS} ${MENAGERIE_TESTS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
