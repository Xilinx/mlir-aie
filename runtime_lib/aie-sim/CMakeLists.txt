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
  include(CTest)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The AIE2/AIE2P register maps are vendored, MIT-licensed, and machine
# readable. Re-deriving them would be both wasted work and a source of drift.
set(AIE_RT_GLOBAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/aie-rt/driver/src/global")

add_library(aie-sim STATIC
  lib/Array.cpp
  lib/CoreEngineLoader.cpp
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

if(NOT WIN32)
  target_link_libraries(aie-sim PUBLIC ${CMAKE_DL_LIBS})
endif()

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

if(NOT AIE_SIM_STANDALONE)
  install(TARGETS aie-sim
          ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/aie-sim/lib)
  install(DIRECTORY include/aiesim
          DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/aie-sim/include)
endif()
