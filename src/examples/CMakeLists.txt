set(MJKDL_EXAMPLES
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
set(MJKDL_EXAMPLES ${MJKDL_EXAMPLES} PARENT_SCOPE)

foreach(ex ${MJKDL_EXAMPLES})
  add_executable(${ex} ${ex}.cpp)
  target_link_libraries(${ex} PRIVATE mjkdl)
  target_compile_options(${ex} PRIVATE -Wall -Wextra)
  target_compile_definitions(${ex} PRIVATE MJKDL_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets")
endforeach()
