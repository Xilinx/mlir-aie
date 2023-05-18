# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Advanced Micro Devices, Inc.

cmake_minimum_required(VERSION 3.21)

project("test lib for ${AIE_RUNTIME_TARGET}")

add_library(test_lib STATIC test_library.cpp)
set_target_properties(test_lib PROPERTIES PUBLIC_HEADER "test_library.h")
target_compile_options(test_lib PRIVATE -fPIC)
target_compile_definitions(test_lib PRIVATE LIBXAIENGINEV2)

target_include_directories(test_lib PRIVATE
    ${LibXAIE_INC_DIR}
)

# copy header files into build area
set(basefile test_library.h)
set(dest ${CMAKE_CURRENT_BINARY_DIR}/../include/${basefile})
add_custom_target(aie-copy-runtime-libs-${basefile} ALL DEPENDS ${dest})
add_custom_command(OUTPUT ${dest}
                  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${basefile} ${dest}
                  DEPENDS ${basefile}
)

install(TARGETS test_lib 
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/lib
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/include
)

# create extra libs supporting simulation
if (DEFINED VITIS_ROOT)
    set(xaienginePath ${VITIS_AIETOOLS_DIR}/include/drivers/aiengine)

    # AIE1 test_lib for aiesimulator
    add_library(test_lib_sim_aie STATIC test_library.cpp)
    target_compile_options(test_lib_sim_aie PRIVATE -fPIC)
    target_compile_definitions(test_lib_sim_aie PRIVATE LIBXAIENGINEV2 __AIESIM__)

    target_include_directories(test_lib_sim_aie PRIVATE
        ${LibXAIE_INC_DIR}
        ${xaienginePath}
    )

    install(TARGETS test_lib_sim_aie 
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/lib
    )

    # AIE2 test_lib for aiesimulator
    add_library(test_lib_sim_aie2 STATIC test_library.cpp)
    target_compile_options(test_lib_sim_aie2 PRIVATE -fPIC)
    target_compile_definitions(test_lib_sim_aie2 PRIVATE LIBXAIENGINEV2 __AIESIM__ __AIEARCH__=20)

    target_include_directories(test_lib_sim_aie2 PRIVATE
        ${LibXAIE_INC_DIR}
        ${xaienginePath}
    )

    install(TARGETS test_lib_sim_aie2 
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/lib
    )
endif(DEFINED VITIS_ROOT)
