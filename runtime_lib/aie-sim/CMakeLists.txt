#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The open AIE array simulator. See docs/AIESimulator.md.
#
# This directory is configurable on its own, so it can be built and tested
# without configuring the rest of mlir-aie:
#
#   cmake -S runtime_lib/aie-sim -B build-sim && cmake --build build-sim
#   ctest --test-dir build-sim
#
# It deliberately depends on nothing but a C++17 compiler and the register
# definitions vendored in third_party/aie-rt.

cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(aie-sim CXX)
  set(AIE_SIM_STANDALONE ON)
endif()

# Unconditionally, not just standalone. In a full mlir-aie build BUILD_TESTING
# may be undefined, and gating the tests on it would silently register none of
# them in exactly the configuration most people build. test/CppTests does the
# same for the same reason.
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The AIE2/AIE2P register maps are vendored, MIT-licensed, and machine
# readable. Re-deriving them would be both wasted work and a source of drift.
set(AIE_RT_GLOBAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/aie-rt/driver/src/global")

add_library(aie-sim STATIC
  lib/Array.cpp
  lib/CoreEngineLoader.cpp
  lib/TileCorePort.cpp
  lib/CoreModule.cpp
  lib/Device.cpp
  lib/Dma.cpp
  lib/EssAbi.cpp
  lib/Lock.cpp
  lib/Memory.cpp
  lib/RegisterFile.cpp
  lib/StreamSwitch.cpp
)

target_include_directories(aie-sim PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(aie-sim PRIVATE ${AIE_RT_GLOBAL_DIR})
set_target_properties(aie-sim PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Put the archive where aiecc's --get-sim looks for it, so a BUILD tree has the
# same layout as an install tree and aiecc works without an install step.
if(NOT AIE_SIM_STANDALONE)
  set_target_properties(aie-sim PROPERTIES ARCHIVE_OUTPUT_DIRECTORY
                        ${CMAKE_BINARY_DIR}/runtime_lib/aie-sim/lib)
endif()

if(NOT WIN32)
  target_link_libraries(aie-sim PUBLIC ${CMAKE_DL_LIBS})
endif()

#===----------------------------------------------------------------------===#
# Optional: link the real vendored aie-rt driver against this model, for
# test/aie_rt_integration_test.cpp. Every other test in this suite configures
# the model by hand-writing registers using this model's OWN layout tables, so
# a misread layout and its test agree with each other and nothing else. This
# tier instead builds aie-rt itself and lets it generate the register traffic
# through the ess_* ABI (docs/AIESimulator.md section 2): if a layout here
# disagrees with aie-rt's, the driver's writes land somewhere this model does
# not expect and the test fails.
#
# Optional and self-skipping, never a hard requirement of this directory's own
# build: a checkout without the third_party/aie-rt submodule, or one where the
# vendored sim de-Vitis-ing patch does not apply, must still configure and
# build every other test.
#===----------------------------------------------------------------------===#
set(AIE_RT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/aie-rt")
set(AIE_RT_PATCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/patches/aie-rt")
set(AIE_RT_DRIVER_SRC "${AIE_RT_ROOT}/driver/src")
set(AIE_SIM_HAVE_AIE_RT FALSE)

if(NOT EXISTS "${AIE_RT_DRIVER_SRC}/xaiengine.h")
  message(STATUS "aie-sim: third_party/aie-rt not present, skipping the "
                 "aie-rt integration test")
else()
  find_package(Git QUIET)
  if(NOT GIT_FOUND)
    message(STATUS "aie-sim: git not found, skipping the aie-rt integration "
                   "test")
  else()
    # apply_aie_rt_vendor_patches() (runtime_lib/xaiengine/aiert.cmake) is the
    # one documented mechanism for touching this submodule, and it is
    # idempotent -- but it FATAL_ERRORs on a patch that neither is already
    # applied nor applies cleanly, which would abort configuring every other
    # test in this directory too. Run its own idempotency check ourselves
    # first, so a conflicting checkout degrades to a skip instead.
    file(GLOB _aie_rt_patches ${AIE_RT_PATCH_DIR}/*.patch)
    set(_aie_rt_patches_ok TRUE)
    foreach(_patch ${_aie_rt_patches})
      execute_process(
        COMMAND ${GIT_EXECUTABLE} apply --reverse --check ${_patch}
        WORKING_DIRECTORY ${AIE_RT_ROOT}
        RESULT_VARIABLE _already_applied
        ERROR_QUIET)
      if(NOT _already_applied EQUAL 0)
        execute_process(
          COMMAND ${GIT_EXECUTABLE} apply --check ${_patch}
          WORKING_DIRECTORY ${AIE_RT_ROOT}
          RESULT_VARIABLE _can_apply
          ERROR_QUIET)
        if(NOT _can_apply EQUAL 0)
          set(_aie_rt_patches_ok FALSE)
        endif()
      endif()
    endforeach()

    if(NOT _aie_rt_patches_ok)
      message(STATUS "aie-sim: third_party/aie-rt sim patch does not apply "
                     "cleanly, skipping the aie-rt integration test")
    else()
      include("${CMAKE_CURRENT_SOURCE_DIR}/../xaiengine/aiert.cmake")
      apply_aie_rt_vendor_patches(${AIE_RT_ROOT} ${AIE_RT_PATCH_DIR})

      enable_language(C)
      find_package(Threads REQUIRED)

      # The flattened staging directory docs/AIESimulator.md section 2
      # describes: aie-rt's headers cross-include each other by bare name
      # ("xaiegbl.h") across what are, in the source tree, different
      # directories, resolving only because the real build (add_aiert_library
      # in aiert.cmake) puts every one of those directories on the include
      # path at once. Reproduce that by copying every header into one
      # directory instead of replicating that whole include list.
      set(AIE_RT_STAGE ${CMAKE_CURRENT_BINARY_DIR}/aie-rt-include)
      file(MAKE_DIRECTORY ${AIE_RT_STAGE}/xaiengine)
      file(GLOB _aie_rt_top_headers ${AIE_RT_DRIVER_SRC}/*.h)
      # Two glob depths: most headers are driver/src/<dir>/*.h, but three
      # (io_backend/ext/xaie_io_common.h and friends) sit one directory
      # deeper. Flattening both depths together is exactly the point.
      file(GLOB _aie_rt_sub_headers
           ${AIE_RT_DRIVER_SRC}/*/*.h ${AIE_RT_DRIVER_SRC}/*/*/*.h)
      file(COPY ${_aie_rt_top_headers} DESTINATION ${AIE_RT_STAGE})
      file(COPY ${_aie_rt_sub_headers} DESTINATION ${AIE_RT_STAGE}/xaiengine)

      file(GLOB AIE_RT_SOURCES
           ${AIE_RT_DRIVER_SRC}/*/*.c ${AIE_RT_DRIVER_SRC}/*/*/*.c)
      add_library(aie_rt_sim STATIC ${AIE_RT_SOURCES})
      # PUBLIC: test/CMakeLists.txt's aie_rt_integration_test.cpp includes
      # "xaiengine.h" itself and needs the same two directories.
      target_include_directories(aie_rt_sim PUBLIC
                                 ${AIE_RT_STAGE} ${AIE_RT_STAGE}/xaiengine)
      target_compile_definitions(aie_rt_sim PRIVATE __AIESIM__)
      set_target_properties(aie_rt_sim PROPERTIES POSITION_INDEPENDENT_CODE ON)
      # xaie_sim.c's XAie_SimIOGetTid calls pthread_self().
      target_link_libraries(aie_rt_sim PUBLIC Threads::Threads)
      # Vendored and unmodified but for the one patch above: not ours to
      # clean up, and this project's own warning flags are not its business.
      target_compile_options(aie_rt_sim PRIVATE -w)

      set(AIE_SIM_HAVE_AIE_RT TRUE)
    endif()
  endif()
endif()

# Reports claimed-register coverage against the vendored register map. A
# review aid rather than a test: much of the map is trace/debug/ECC this model
# deliberately omits, so there is no number to assert.
add_executable(register_coverage tools/register_coverage.cpp)
target_link_libraries(register_coverage PRIVATE aie-sim)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

if(NOT AIE_SIM_STANDALONE)
  install(TARGETS aie-sim
          ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/aie-sim/lib)
  install(DIRECTORY include/aiesim
          DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/aie-sim/include)
endif()
