#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Xilinx Inc.

# Stuff into the build area:
add_custom_target(aie-kernels ALL)

function(add_aie_kernels arch)
  if(DEFINED VITIS_AIETOOLS_DIR)
    if(EXISTS "${PROJECT_SOURCE_DIR}/tools/chess-clang/xchesscc_wrapper")
        add_custom_target(${arch}_kernels ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mm.o)
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mm.o
                      COMMAND ${PROJECT_SOURCE_DIR}/tools/chess-clang/xchesscc_wrapper ${arch}
                        -aietools ${VITIS_AIETOOLS_DIR}
                        -I ${VITIS_AIETOOLS_DIR}/include
                        -c ${CMAKE_CURRENT_SOURCE_DIR}/mm.cc
                        -o ${CMAKE_CURRENT_BINARY_DIR}/mm.o
                      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/mm.cc)
        add_dependencies(aie-kernels ${arch}_kernels)
    
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mm.o DESTINATION ${CMAKE_INSTALL_PREFIX}/aie_kernels/)
    else()
      message(STATUS "Could not find xchesscc_wrapper inside mlir-aie's build directory")
    endif()
  else()
      message(STATUS "Tool 'xchesscc' not found. Skipping compilation of microkernel .o files.")
  endif()

endfunction()

add_aie_kernels(AIE2)