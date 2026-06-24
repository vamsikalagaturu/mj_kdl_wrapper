set(EXAMPLES
  ex_gravity_comp
  ex_pos_ctrl
  ex_vel_ctrl
  ex_impedance
  ex_pick
  ex_table_pick_place
  ex_rnea_pick_place
  ex_achd_table_slide
  ex_achd_pick_place
  ex_table_pour
  ex_table_scene
  ex_dual_arm
  ex_record
)

foreach(ex ${EXAMPLES})
  add_executable(${ex} ${ex}.cpp)
  target_link_libraries(${ex} PRIVATE mj_kdl_wrapper)
  target_compile_options(${ex} PRIVATE -Wall -Wextra)
endforeach()

# The ACHD examples use the fork's fixed-joint Vereshchagin solver. It is
# compiled into liborocos-kdl (linked transitively via mj_kdl_wrapper) and its
# public header ships under kdl/, so no extra source or build-order wiring is
# needed - including the orocos_kdl_ep ExternalProject, which does not exist
# when KDL is consumed as a package (MJ_KDL_OROCOS_KDL_FROM_PACKAGE/INSTALL_DIR).
