set(MJ_KDL_EXAMPLES
  ex_gravity_comp
  ex_joint_ctrl
  ex_admittance_ft
  ex_table_pick_place
  ex_rnea_pick_place
  ex_achd_table_slide
  ex_achd_pick_place
  ex_table_pour
)
# test/ registers each one as a headless test.
set(MJ_KDL_EXAMPLES ${MJ_KDL_EXAMPLES} PARENT_SCOPE)

foreach(ex ${MJ_KDL_EXAMPLES})
  add_executable(${ex} ${ex}.cpp)
  target_link_libraries(${ex} PRIVATE mj_kdl_wrapper)
  target_compile_options(${ex} PRIVATE -Wall -Wextra)
  if(TARGET mj_kdl_cache_assets)
    add_dependencies(${ex} mj_kdl_cache_assets)
  endif()
endforeach()
