configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TEST_DEPENDS
  FileCheck count not
  copy-python
  aie-opt
  aie-translate
  )

add_lit_testsuite(check-aie "Running the aie regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  )
set_target_properties(check-aie PROPERTIES FOLDER "Tests")

add_lit_testsuites(AIE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TEST_DEPENDS})
