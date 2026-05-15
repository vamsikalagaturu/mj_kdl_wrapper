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

set(_OROCOS_KDL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../orocos_kinematics_dynamics/orocos_kdl/src")
target_sources(ex_achd_pick_place PRIVATE
  "${_OROCOS_KDL_SRC}/chainhdsolver_vereshchagin_fixed_joint.cpp"
)
target_include_directories(ex_achd_pick_place PRIVATE "${_OROCOS_KDL_SRC}")
target_sources(ex_achd_table_slide PRIVATE
  "${_OROCOS_KDL_SRC}/chainhdsolver_vereshchagin_fixed_joint.cpp"
)
target_include_directories(ex_achd_table_slide PRIVATE "${_OROCOS_KDL_SRC}")
