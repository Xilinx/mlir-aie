# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Advanced Micro Devices, Inc.

cmake_minimum_required(VERSION 3.21)

project("test lib for ${AIE_RUNTIME_TARGET}")

add_library(test_lib STATIC test_library.cpp)
set(TEST_LIB_PUBLIC_HEADERS
    test_library.h
    target.h
)
set_target_properties(test_lib PROPERTIES PUBLIC_HEADER "${TEST_LIB_PUBLIC_HEADERS}")
target_compile_options(test_lib PRIVATE -fPIC)

target_include_directories(test_lib PRIVATE
    ${LibXAIE_INC_DIR}
)

# copy header and source files into build area
set(headers target.h test_library.h memory_allocator.h)
foreach(basefile ${headers})
    set(dest ${CMAKE_CURRENT_BINARY_DIR}/../include/${basefile})
    add_custom_target(aie-copy-runtime-libs-${basefile} ALL DEPENDS ${dest})
    add_custom_command(OUTPUT ${dest}
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${basefile} ${dest}
                    DEPENDS ${basefile}
    )
endforeach()

set(files test_library.cpp)
foreach(basefile ${files})
    set(dest ${CMAKE_CURRENT_BINARY_DIR}/../src/${basefile})
    add_custom_target(aie-copy-runtime-libs-${basefile} ALL DEPENDS ${dest})
    add_custom_command(OUTPUT ${dest}
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${basefile} ${dest}
                    DEPENDS ${basefile}
    )
endforeach()

install(TARGETS test_lib 
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/lib
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/include
)
install(FILES test_library.cpp DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/src)

set(xaienginePath ${VITIS_AIETOOLS_DIR}/include/drivers/aiengine)
# Memory Allocator
add_library(memory_allocator_ion STATIC memory_allocator_ion.cpp)
set_target_properties(memory_allocator_ion PROPERTIES PUBLIC_HEADER "memory_allocator.h")
find_program(UNAME_EXEC uname)
execute_process(COMMAND ${UNAME_EXEC} -r OUTPUT_VARIABLE KERNEL_RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE)
find_path(LINUX_HEADERS_PATH NAMES "linux/dma-buf.h" PATHS "/usr/src/kernels/${KERNEL_RELEASE}/include" REQUIRED)
target_include_directories(memory_allocator_ion PRIVATE
        ${LINUX_HEADERS_PATH}/uapi
        ${LINUX_HEADERS_PATH}
        ${LibXAIE_INC_DIR}
)
target_compile_options(memory_allocator_ion PRIVATE -fPIC)
target_compile_definitions(memory_allocator_ion PRIVATE)

install(TARGETS memory_allocator_ion
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/lib
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/include
)

if (VITIS_ROOT)
    add_library(memory_allocator_sim_aie STATIC memory_allocator.cpp)
    target_compile_options(memory_allocator_sim_aie PRIVATE -fPIC)
    target_compile_definitions(memory_allocator_sim_aie PRIVATE __AIESIM__)
    target_include_directories(memory_allocator_sim_aie PRIVATE ${xaienginePath})

    install(TARGETS memory_allocator_sim_aie
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/lib
    )
else()
    message(STATUS "Vitis not found, simulation memory allocator will not be built.")
endif()
