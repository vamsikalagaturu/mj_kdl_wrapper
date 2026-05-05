set(EXAMPLES
  ex_init
  ex_gravity_comp
  ex_pos_ctrl
  ex_vel_ctrl
  ex_gripper
  ex_impedance
  ex_pick
  ex_table_pick_place
  ex_table_scene
  ex_dual_arm
  ex_record
)

foreach(ex ${EXAMPLES})
  add_executable(${ex} ${ex}.cpp)
  target_link_libraries(${ex} PRIVATE mj_kdl_wrapper)
  target_compile_options(${ex} PRIVATE -Wall -Wextra)
endforeach()
