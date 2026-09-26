# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# parameters
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

# -----------------------------------------------------------------------------
# Sweep selection
# -----------------------------------------------------------------------------
# This CMakeLists is shared by all three sweep subdirs: each subdir Makefile
# points build_host_exe at ${srcdir}/.. (i.e. here) with ../test.cpp, so the
# host-only make flow must keep working untouched. The design-side knobs below
# only matter for the CMake+ctest flow (AIE_BUILD_DESIGN=ON); build_host_exe
# passes -DAIE_BUILD_DESIGN=OFF and supplies IN1_SIZE/OUT_SIZE/DTYPE directly.
set(AIE_DEVICE npu CACHE STRING "NPU device family for the example (npu|npu2)")
set(DESIGN_DIR single_core_designs CACHE STRING
    "Sweep subdir holding the design (single_core_designs|single_column_designs|multi_column_designs)")
set(VARIANT "" CACHE STRING
    "Design variant: shared|chained|memtile (single_column) or col_wise|row_wise (multi_column)")
set(AIE_DTYPE i32 CACHE STRING "Element type passed to the design (i32|bf16)")

# Resolve the design .py name and the per-subdir default in1_size, mirroring
# each subdir Makefile's aie_py_src / in1_size.
if(DESIGN_DIR STREQUAL "single_core_designs")
  set(_py "vector_reduce_max.py")
  set(_default_in1_size 8192)
  if(VARIANT)
    message(FATAL_ERROR "single_core_designs has no VARIANT (got '${VARIANT}')")
  endif()
elseif(DESIGN_DIR STREQUAL "single_column_designs")
  if(NOT VARIANT)
    set(VARIANT shared)
  endif()
  if(NOT VARIANT MATCHES "^(shared|chained|memtile)$")
    message(FATAL_ERROR "single_column_designs VARIANT must be shared|chained|memtile (got '${VARIANT}')")
  endif()
  set(_py "vector_reduce_max_${VARIANT}.py")
  set(_default_in1_size 524288)
elseif(DESIGN_DIR STREQUAL "multi_column_designs")
  if(NOT VARIANT)
    set(VARIANT col_wise)
  endif()
  if(NOT VARIANT MATCHES "^(col_wise|row_wise)$")
    message(FATAL_ERROR "multi_column_designs VARIANT must be col_wise|row_wise (got '${VARIANT}')")
  endif()
  set(_py "${VARIANT}_vector_reduce_max.py")
  set(_default_in1_size 524288)
else()
  message(FATAL_ERROR "Unknown DESIGN_DIR '${DESIGN_DIR}' "
                      "(single_core_designs|single_column_designs|multi_column_designs)")
endif()

set(IN1_SIZE ${_default_in1_size} CACHE STRING "in1 buffer size (bytes)")
set(OUT_SIZE 4 CACHE STRING "out buffer size (bytes)")

# DTYPE is the C++ type name baked into test.cpp; the Makefiles pass it directly
# (-DDTYPE=std::int32_t / test_utils::bfloat16_t). In the CMake flow nothing sets
# it, so derive it from AIE_DTYPE -- the design arg and the host define must
# agree, a mismatch is silent corruption rather than a clean failure.
set(DTYPE "" CACHE STRING "C++ element type for the host testbench (set by the make flow)")
if(NOT DTYPE)
  if(AIE_DTYPE STREQUAL "bf16")
    set(DTYPE test_utils::bfloat16_t)
  elseif(AIE_DTYPE STREQUAL "i32")
    set(DTYPE std::int32_t)
  else()
    message(FATAL_ERROR "AIE_DTYPE must be i32 or bf16 (got '${AIE_DTYPE}')")
  endif()
endif()

add_executable(${currentTarget} test.cpp)

target_compile_definitions(${currentTarget} PUBLIC
    IN1_SIZE=${IN1_SIZE}
    OUT_SIZE=${OUT_SIZE}
    DTYPE=${DTYPE}
)

target_include_directories (${currentTarget} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils
    ${XRT_INC_DIR}
)

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
)

target_link_test_utils(${currentTarget})

# Build the design and register the NPU run; -DAIE_DEVICE=npu2 targets Strix.
# The design .py lives in the sweep subdir, not next to this CMakeLists, hence
# PY_DIR. Args mirror the subdir Makefile's ${aieargs}.
add_aie_design(TARGET ${currentTarget} PY ${_py} DEVICE ${AIE_DEVICE}
               PY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${DESIGN_DIR}"
               ARGS -i1s ${IN1_SIZE} -os ${OUT_SIZE} --dtype ${AIE_DTYPE})
add_dependencies(${currentTarget} ${currentTarget}_xclbin)
# Mirrors the Makefile `run:` recipe: ./exe -x final.xclbin -i insts.bin -k MLIR_AIE
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} EXE ${currentTarget})
