set(EXAMPLES
  ex_init
  ex_gravity_comp
  ex_velocity
  ex_gripper
  ex_impedance
  ex_pick
  ex_dual_arm
)

foreach(ex ${EXAMPLES})
  add_executable(${ex} ${ex}.cpp)
  target_link_libraries(${ex} PRIVATE mj_kdl_wrapper)
  target_compile_options(${ex} PRIVATE -Wall -Wextra)
endforeach()
