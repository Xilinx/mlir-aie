# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2023, Advanced Micro Devices, Inc.

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
)

add_lit_testsuite(check-tutorials "Running the aie tutorials tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS "-sv --timeout 300 --time-tests"
)

set_target_properties(check-tutorials PROPERTIES FOLDER "Tutorials")

add_lit_testsuites(TUTORIALS ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${TEST_DEPENDS} ARGS "-sv --timeout 300 --time-tests")
