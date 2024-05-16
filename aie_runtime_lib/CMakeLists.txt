#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

# Stuff into the build area:
add_custom_target(aie-runtime-libs ALL)

function(add_aie_runtime_libs arch) 
  add_custom_target(${arch}_me_basic ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o)
  if(DEFINED VITIS_ROOT)
      # Compile me_basic.o
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o
                      COMMAND ${VITIS_XCHESSCC} -p me -P ${VITIS_${arch}_INCLUDE_DIR}
                      -C Release
                      -I ${VITIS_${arch}_INCLUDE_DIR}/runtime/include/
                      -d -c ${CMAKE_CURRENT_SOURCE_DIR}/me_basic.c
                      -o ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o
                      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/me_basic.c)
  else()
      # Exists in the source tree.. just copy it.
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o
                      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/me_basic.o
                      ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o
                      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/me_basic.o)
  endif()
  add_dependencies(aie-runtime-libs ${arch}_me_basic)

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o DESTINATION ${CMAKE_INSTALL_PREFIX}/aie_runtime_lib/${arch})
  
  # Precompile the intrinsic wrappers.  
  if(DEFINED VITIS_ROOT)
    add_custom_target(${arch}_chess_intrinsic_wrapper ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/chess_intrinsic_wrapper.ll)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/chess_intrinsic_wrapper.ll
            COMMAND xchesscc_wrapper ${arch} -p me
            -I ${VITIS_${arch}_INCLUDE_DIR}/runtime/include/
            -d -c ${CMAKE_CURRENT_SOURCE_DIR}/chess_intrinsic_wrapper.cpp
            -f +f
            -o ${CMAKE_CURRENT_BINARY_DIR}/chess_intrinsic_wrapper.ll
            COMMAND sed -i s/^target.*//
            ${CMAKE_CURRENT_BINARY_DIR}/chess_intrinsic_wrapper.ll
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/chess_intrinsic_wrapper.cpp)
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

add_subdirectory(AIE)
add_subdirectory(AIE2)


