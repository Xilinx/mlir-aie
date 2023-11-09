if(ENABLE_CHESS_TESTS)
set(CONFIG_ENABLE_CHESS_TESTS 1)
else()
set(CONFIG_ENABLE_CHESS_TESTS 0)
endif()
if(ENABLE_BOARD_TESTS)
set(CONFIG_ENABLE_BOARD_TESTS 1)
else()
set(CONFIG_ENABLE_BOARD_TESTS 0)
endif()
if(LibXAIE_FOUND)
set(CONFIG_HAS_LIBXAIE 1)
else()
set(CONFIG_HAS_LIBXAIE 0)
endif()

set(ENABLE_PYTHON_TESTS 1)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TEST_DEPENDS
  FileCheck count not
  aiecc.py
  aie-opt
  aie-translate
  AIEPythonModules
)

add_lit_testsuite(check-aie "Running the aie regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS "-sv --timeout 600 --time-tests --show-unsupported"
)

set_target_properties(check-aie PROPERTIES FOLDER "Tests")

add_lit_testsuites(AIE ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${TEST_DEPENDS} ARGS "-sv --timeout 600 --time-tests --show-unsupported")
