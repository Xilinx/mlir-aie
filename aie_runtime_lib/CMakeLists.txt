#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

# Stuff into the build area:
add_custom_target(aie-runtime-libs ALL)

function(add_aie_runtime_libs arch) 
  # Precompile the intrinsic wrappers.  
  if(DEFINED VITIS_AIETOOLS_DIR)
    string(TOLOWER ${arch} aiearch)
    add_custom_target(${arch}_chess_intrinsic_wrapper ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/chess_intrinsic_wrapper.ll)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/chess_intrinsic_wrapper.ll
            COMMAND ${PROJECT_BINARY_DIR}/bin/xchesscc_wrapper ${arch}
            -aietools ${AIETOOLS_DIR}
            -c ${CMAKE_CURRENT_SOURCE_DIR}/chess_intrinsic_wrapper.cpp
            -d # Disassemble output
            -f # Use LLVM frontend
            +f # ... and emit LLVMIR
            -o ${CMAKE_CURRENT_BINARY_DIR}/chess_intrinsic_wrapper.ll
            COMMAND sed -i s/^target.*//
            ${CMAKE_CURRENT_BINARY_DIR}/chess_intrinsic_wrapper.ll
            DEPENDS
            xchesscc_wrapper
            ${CMAKE_CURRENT_SOURCE_DIR}/chess_intrinsic_wrapper.cpp)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/chess_intrinsic_wrapper.ll DESTINATION ${CMAKE_INSTALL_PREFIX}/aie_runtime_lib/${arch})
    add_dependencies(aie-runtime-libs ${arch}_chess_intrinsic_wrapper)
  endif()

  set(INSTALLS
      lut_based_ops.cpp
      lut_based_ops.h
      vec_math.h)

  foreach(file ${INSTALLS})
      add_custom_target(aie-copy-${arch}-runtime-libs-${file} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${file})
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file}
                      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
                      ${CMAKE_CURRENT_BINARY_DIR}/${file}
                      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
      add_dependencies(aie-runtime-libs aie-copy-${arch}-runtime-libs-${file})
  endforeach()

  install(FILES ${INSTALLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/aie_runtime_lib/${arch})

  add_subdirectory(aiesim)

endfunction()

# Compile AIE runtime libs if component found
if(AIETools_AIE_FOUND)
  add_subdirectory(AIE)
endif()
if(AIETools_AIE2_FOUND)
  add_subdirectory(AIE2)
endif()


