# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Advanced Micro Devices, Inc.

cmake_minimum_required(VERSION 3.20.1)

project("xaiengine lib for ${AIE_RUNTIME_TARGET}")
include("aiert.cmake")

if (${CMAKE_CROSSCOMPILING})
    message("Building xaiengine for ${AIE_RUNTIME_TARGET} from Vitis at ${VITIS_ROOT}.")
    add_subdirectory(lib)

else()
    message("Copying xaiengine for ${AIE_RUNTIME_TARGET} from ${AIERT_LIBS}.") 
    add_aiert_headers(xaiengine
                        ${AIERT_INCLUDE_DIR}
                        ${CMAKE_CURRENT_BINARY_DIR}/include
                        ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/xaiengine/include)

    # copy library file to build area
    set(libs
        ${CMAKE_CURRENT_BINARY_DIR}/lib/libxaiengine.so
        ${CMAKE_CURRENT_BINARY_DIR}/lib/libxaiengine.so.3)
    add_custom_target(aie-copy-runtime-libs ALL DEPENDS ${libs})
    foreach(lib ${libs})
        add_custom_command(OUTPUT ${lib} COMMAND ${CMAKE_COMMAND} -E copy ${AIERT_LIBS} ${lib})
    endforeach()

    # install library and headers
    install(FILES ${libs} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/xaiengine/lib)
endif()

add_subdirectory(cdo)