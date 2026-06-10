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

# The ACHD examples use the secorolab fork's fixed-joint Vereshchagin solver,
# which is not part of liborocos-kdl. It is downloaded by the orocos_kdl_ep
# ExternalProject at build time, so the source is marked GENERATED and the
# examples depend on that step.
set(_OROCOS_KDL_SRC "${_MJ_KDL_OROCOS_SRC}/orocos_kdl/src")
set(_ACHD_SOLVER_SRC "${_OROCOS_KDL_SRC}/chainhdsolver_vereshchagin_fixed_joint.cpp")
set_source_files_properties("${_ACHD_SOLVER_SRC}" PROPERTIES GENERATED TRUE)

foreach(ex ex_achd_pick_place ex_achd_table_slide)
  target_sources(${ex} PRIVATE "${_ACHD_SOLVER_SRC}")
  target_include_directories(${ex} PRIVATE "${_OROCOS_KDL_SRC}")
  add_dependencies(${ex} orocos_kdl_ep)
endforeach()
