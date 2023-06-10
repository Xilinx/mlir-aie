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

target_include_directories(test_lib PRIVATE
    ${LibXAIE_INC_DIR}
)

# copy header and source files into build area
set(basefile test_library.h)
set(dest ${CMAKE_CURRENT_BINARY_DIR}/../include/${basefile})
add_custom_target(aie-copy-runtime-libs-${basefile} ALL DEPENDS ${dest})
add_custom_command(OUTPUT ${dest}
                  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${basefile} ${dest}
                  DEPENDS ${basefile}
)

set(basefile test_library.cpp)
set(dest ${CMAKE_CURRENT_BINARY_DIR}/../src/${basefile})
add_custom_target(aie-copy-runtime-libs-${basefile} ALL DEPENDS ${dest})
add_custom_command(OUTPUT ${dest}
                  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${basefile} ${dest}
                  DEPENDS ${basefile}
)

install(TARGETS test_lib 
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/lib
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/include
)
install(FILES test_library.cpp DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/src)

set(xaienginePath ${VITIS_AIETOOLS_DIR}/include/drivers/aiengine)
# Memory Allocator
add_library(memory_allocator_ion STATIC memory_allocator_ion.cpp)
target_compile_options(memory_allocator_ion PRIVATE -fPIC)
target_compile_definitions(memory_allocator_ion PRIVATE)

install(TARGETS memory_allocator_ion
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/lib
)

add_library(memory_allocator_sim_aie STATIC memory_allocator.cpp)
target_compile_options(memory_allocator_sim_aie PRIVATE -fPIC)
target_compile_definitions(memory_allocator_sim_aie PRIVATE __AIESIM__)
target_include_directories(memory_allocator_sim_aie PRIVATE ${xaienginePath})

install(TARGETS memory_allocator_sim_aie
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/lib
)
