# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

cmake_minimum_required(VERSION 3.30)
project("test lib for ${AIE_RUNTIME_TARGET}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

function(add_runtime_pic_flag target_name)
  if (NOT WIN32)
    target_compile_options(${target_name} PRIVATE -fPIC)
  endif()
endfunction()

find_package(XRT QUIET)

set(BUILD_TEST_LIBRARY ON)
set(BUILD_TEST_UTILS_LIBRARY ON)
set(STAGE_TEST_UTILS_PAYLOAD ON)
if (WIN32)
  set(BUILD_TEST_LIBRARY OFF)
endif()

# test_lib library
if (BUILD_TEST_LIBRARY)
  add_library(test_lib STATIC test_library.cpp)
  set(TEST_LIB_PUBLIC_HEADERS
      test_library.h
      target.h
      cxxopts.hpp
  )
  set_target_properties(test_lib PROPERTIES PUBLIC_HEADER "${TEST_LIB_PUBLIC_HEADERS}")
  add_runtime_pic_flag(test_lib)

  target_include_directories(test_lib PRIVATE
      ${LibXAIE_INC_DIR}
  )
endif()

# test_utils library
if (BUILD_TEST_UTILS_LIBRARY)
  add_library(test_utils STATIC test_utils.cpp)
  set_target_properties(test_utils PROPERTIES PUBLIC_HEADER "test_utils.h;xrt_test_wrapper.h;cxxopts.hpp;parameter_scratchpad.h")
  add_runtime_pic_flag(test_utils)

  # XRT headers on Windows pull in boost/any.hpp which is not typically
  # available.  Skip XRT integration there; the core helpers (load_instr_*,
  # nearly_equal, etc.) are still usable without it.
  if (${XRT_FOUND} AND NOT WIN32)
    target_compile_options(test_utils PRIVATE -DTEST_UTILS_USE_XRT)
    target_include_directories(test_utils PRIVATE ${XRT_INCLUDE_DIR})
  endif()
  target_include_directories(test_utils PRIVATE
      ${LibXAIE_INC_DIR}
  )
endif()

# copy the headers used by the enabled runtime helpers into build area
set(headers cxxopts.hpp)
if (BUILD_TEST_LIBRARY)
  list(APPEND headers target.h test_library.h memory_allocator.h)
endif()
if (STAGE_TEST_UTILS_PAYLOAD)
  list(APPEND headers test_utils.h xrt_test_wrapper.h parameter_scratchpad.h)
endif()
foreach(basefile ${headers})
    set(dest ${CMAKE_CURRENT_BINARY_DIR}/../include/${basefile})
    add_custom_target(aie-copy-runtime-libs-${basefile} ALL DEPENDS ${dest})
    add_custom_command(OUTPUT ${dest}
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${basefile} ${dest}
                    DEPENDS ${basefile}
    )
endforeach()

# copy the sources used by the enabled runtime helpers into build area
set(files)
if (BUILD_TEST_LIBRARY)
  list(APPEND files test_library.cpp)
endif()
if (STAGE_TEST_UTILS_PAYLOAD)
  list(APPEND files test_utils.cpp)
endif()
foreach(basefile ${files})
    set(dest ${CMAKE_CURRENT_BINARY_DIR}/../src/${basefile})
    add_custom_target(aie-copy-runtime-libs-${basefile} ALL DEPENDS ${dest})
    add_custom_command(OUTPUT ${dest}
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${basefile} ${dest}
                    DEPENDS ${basefile}
    )
endforeach()

if (BUILD_TEST_LIBRARY)
  install(TARGETS test_lib 
      ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/lib
      PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/include
  )
  install(FILES test_library.cpp DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/src)
endif()
if (BUILD_TEST_UTILS_LIBRARY)
  install(TARGETS test_utils
      ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/lib
      PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/include
  )
elseif (STAGE_TEST_UTILS_PAYLOAD)
  install(FILES test_utils.h xrt_test_wrapper.h cxxopts.hpp parameter_scratchpad.h
      DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/include
  )
endif()

#
# Memory Allocator libraries
#

# ion / dma-buf allocator is Linux-only.
if (NOT WIN32)
  add_library(memory_allocator_ion STATIC memory_allocator_ion.cpp)
  set(ION_PUBLIC_HEADERS
      memory_allocator.h
      target.h
  )
  set_target_properties(memory_allocator_ion PROPERTIES PUBLIC_HEADER "${ION_PUBLIC_HEADERS}")
  find_program(UNAME_EXEC uname)
  execute_process(COMMAND ${UNAME_EXEC} -r OUTPUT_VARIABLE KERNEL_RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_path(LINUX_HEADERS_PATH NAMES "linux/dma-buf.h" PATHS "/usr/src/kernels/${KERNEL_RELEASE}/include" REQUIRED)
  target_include_directories(memory_allocator_ion PRIVATE
          ${LINUX_HEADERS_PATH}/uapi
          ${LINUX_HEADERS_PATH}
          ${LibXAIE_INC_DIR}
  )
  add_runtime_pic_flag(memory_allocator_ion)
  target_compile_definitions(memory_allocator_ion PRIVATE)

  install(TARGETS memory_allocator_ion
      ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/lib
      PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/include
  )
endif()

message("LibXAIE_INC_DIR is at ${LibXAIE_INC_DIR} for ${AIE_RUNTIME_TARGET}")

# simulation memory allocator library
if (NOT WIN32)
  add_library(memory_allocator_sim_aie STATIC memory_allocator.cpp)
  add_runtime_pic_flag(memory_allocator_sim_aie)
  target_compile_definitions(memory_allocator_sim_aie PRIVATE __AIESIM__)
  target_include_directories(memory_allocator_sim_aie PRIVATE ${LibXAIE_INC_DIR})

  install(TARGETS memory_allocator_sim_aie
      ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/test_lib/lib
  )
endif()
