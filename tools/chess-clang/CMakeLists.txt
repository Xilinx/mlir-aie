#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

if(NOT DEFINED CMAKE_CROSSCOMPILE)
  set(SCRIPT_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/bin)
  # These files are only useful on host x86 compiles.
  configure_file(xchesscc_wrapper.in ${PROJECT_BINARY_DIR}/bin/xchesscc_wrapper @ONLY)
  install(PROGRAMS ${PROJECT_BINARY_DIR}/bin/xchesscc_wrapper DESTINATION ${SCRIPT_INSTALL_PATH})

  set(FILES
    chess-clang)

  set(TARGETS ${FILES})
  list(TRANSFORM TARGETS PREPEND ${PROJECT_BINARY_DIR}/bin/)

  # stuff into the build directory
  add_custom_target(copy-scripts ALL DEPENDS ${TARGETS})
  foreach(file ${FILES})
  add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bin/${file}
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
                    ${PROJECT_BINARY_DIR}/bin/${file}
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
  endforeach()

  # install too!
  foreach(file ${FILES})
    install(PROGRAMS ${file} DESTINATION ${SCRIPT_INSTALL_PATH})
  endforeach()
endif()