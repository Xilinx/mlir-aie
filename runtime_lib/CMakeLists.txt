#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

include(ExternalProject)

# copy libc and libm from Vitis
function(add_aie_runtime_libs arch) 
  if(DEFINED VITIS_ROOT)
    add_custom_target(aie-copy-${arch}-runtime-libs-libc ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libc.a)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libc.a
                    COMMAND ${CMAKE_COMMAND} -E copy ${VITIS_${arch}_LIBC}
                    ${CMAKE_CURRENT_BINARY_DIR}/libc.a
                    DEPENDS ${VITIS_${arch}_LIBC})
    add_dependencies(runtime-libs aie-copy-${arch}-runtime-libs-libc )

    add_custom_target(aie-copy-${arch}-runtime-libs-libsoftfloat ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libsoftfloat.a)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libsoftfloat.a
                    COMMAND ${CMAKE_COMMAND} -E copy ${VITIS_${arch}_LIBSOFTFLOAT}
                    ${CMAKE_CURRENT_BINARY_DIR}/libsoftfloat.a
                    DEPENDS ${VITIS_${arch}_LIBSOFTFLOAT})
    add_dependencies(runtime-libs aie-copy-${arch}-runtime-libs-libsoftfloat )

    add_custom_target(aie-copy-${arch}-runtime-libs-libm ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libm.a)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libm.a
                    COMMAND ${CMAKE_COMMAND} -E copy ${VITIS_${arch}_LIBM}
                    ${CMAKE_CURRENT_BINARY_DIR}/libm.a
                    DEPENDS ${VITIS_${arch}_LIBM})
    add_dependencies(runtime-libs aie-copy-${arch}-runtime-libs-libm )
    install(FILES ${VITIS_${arch}_LIBC} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${arch})
    install(FILES ${VITIS_${arch}_LIBM} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${arch})
    install(FILES ${VITIS_${arch}_LIBSOFTFLOAT} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${arch})
  else()
      message(WARNING "No source of libm/libc... Hope you don't need it!")
  endif()

  # Extract me_basic.o from libme.a, which is shipped with Vitis.
  add_custom_target(${arch}_me_basic ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o)
  if(DEFINED VITIS_ROOT)
      # Compile me_basic.o
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o
                      COMMAND ${VITIS_XCHESSCC} -p me -P ${VITIS_${arch}_INCLUDE_DIR}
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
  add_dependencies(runtime-libs ${arch}_me_basic)

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${arch})

endfunction()

set(INSTALLS
    test_library.cpp
    test_library.h
    chess_intrinsic_wrapper.cpp)

# Stuff into the build area:
add_custom_target(runtime-libs ALL)

foreach(file ${INSTALLS})
    add_custom_target(aie-copy-runtime-libs-${file} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${file})
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file}
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
                    ${CMAKE_CURRENT_BINARY_DIR}/${file}
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    add_dependencies(runtime-libs aie-copy-runtime-libs-${file} )
endforeach()

# Install too
install(FILES ${INSTALLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib)

add_subdirectory(aiesim)
add_subdirectory(AIE)
add_subdirectory(AIE2)

if(DEFINED VITIS_ROOT)
ExternalProject_Add(xaiengine
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/xaiengineTmp
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/xaiengine
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/xaiengine
  CMAKE_ARGS
    -DCMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAINFILES_PATH}/toolchain_vitis_crosscomp_arm.cmake
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  )
endif()
