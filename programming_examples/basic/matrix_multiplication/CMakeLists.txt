# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# parameters
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built
# -Dsubdir:      Which matmul variant to build (single_core|whole_array|
#                matrix_vector|cascade). Also selects the design .py.
#
# This one CMakeLists is shared by every matmul sibling, exactly as in the make
# flow: makefile-common's build_host_exe configures this directory with
# `-Dsubdir=<name> -DAIE_BUILD_DESIGN=OFF` to produce only the host binary.
# The run_*_cmake.lit tests configure the same directory with AIE_BUILD_DESIGN
# left ON, so CMake additionally JITs the design and registers the ctest.

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

add_executable(${currentTarget}
    ${subdir}/test.cpp
)

target_include_directories (${currentTarget} PUBLIC ${XRT_INC_DIR})

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
)

target_link_test_utils(${currentTarget})

# -----------------------------------------------------------------------------
# CMake-driven design build + NPU run (run_cmake.lit / run_strix_cmake.lit)
# -----------------------------------------------------------------------------
# Everything below is the CMake equivalent of the per-sibling Makefile +
# makefile-common: the ${aieargs} passed to <subdir>/<subdir>.py, the
# -DDTYPE_* host defines derived from dtype_in/dtype_out, and the `run:`
# recipe registered as a ctest.
#
# It is deliberately all inside `if(AIE_BUILD_DESIGN)`:
#   * makefile-common's ${targetname}.exe rule configures this same directory
#     to produce ONLY the host binary, and supplies the -DDTYPE_* defines
#     itself through CXXFLAGS. Adding them again here would either duplicate
#     the JIT or, worse, fight the CXXFLAGS values and silently mis-type the
#     host buffers. The make rule passes -DAIE_BUILD_DESIGN=OFF, so this whole
#     block is skipped and the make flow behaves exactly as before.
#   * the run_*_cmake.lit tests leave AIE_BUILD_DESIGN at its default ON.
if(AIE_BUILD_DESIGN)

set(AIE_DEVICE npu CACHE STRING "NPU device family for the example (npu|npu2)")

# --- Per-sibling defaults, mirroring each subdir's Makefile ------------------
if(subdir STREQUAL "matrix_vector")
  # matrix_vector/Makefile pins these (plain `=`, not `?=`) because the
  # vectorized mv kernel is built for the (m=32, k=32) tile.
  set(_def_M 288)
  set(_def_K 288)
  set(_def_N 1)
else()
  set(_def_M 512)
  set(_def_K 512)
  set(_def_N 512)
endif()

set(M ${_def_M} CACHE STRING "GEMM M")
set(K ${_def_K} CACHE STRING "GEMM K")
set(N ${_def_N} CACHE STRING "GEMM N")
set(TILE_M 32 CACHE STRING "AIE tile rows (Makefile `m`)")
set(TILE_K 32 CACHE STRING "AIE tile depth (Makefile `k`)")
set(TILE_N 32 CACHE STRING "AIE tile cols (Makefile `n`)")
set(N_AIE_COLS 4 CACHE STRING "Number of AIE array columns (Makefile `n_aie_cols`)")
set(B_COL_MAJ 0 CACHE STRING "B matrix is column-major (0|1)")
set(C_COL_MAJ 0 CACHE STRING "C matrix is column-major (0|1)")
set(USE_SCALAR 0 CACHE STRING "Use the scalar (non-vectorized) kernels (0|1)")
set(DTYPE_IN i16 CACHE STRING "Input element type (bf16|i8|i16)")
set(DTYPE_OUT i32 CACHE STRING "Output element type (bf16|i8|i16|i32|f32)")
set(EMULATE_BF16_MMUL_WITH_BFP16 0 CACHE STRING
    "Emulate the bf16 matmul with bfp16 (0|1)")
set(USE_CHESS 0 CACHE STRING "Build the kernels with xchesscc instead of Peano (0|1)")

# --- dtype_in / dtype_out -> C++ type names ----------------------------------
# Exact transcription of the dtype mapping block in makefile-common. The design
# args and these host defines MUST agree; a mismatch corrupts the buffers
# silently instead of failing, so unknown values are a hard error here rather
# than the empty `-DDTYPE_IN=` the Makefile would emit.
if(DTYPE_IN STREQUAL "bf16")
  set(_dtype_in_cpp test_utils::bfloat16_t)
elseif(DTYPE_IN STREQUAL "i16")
  set(_dtype_in_cpp int16_t)
elseif(DTYPE_IN STREQUAL "i8")
  set(_dtype_in_cpp int8_t)
else()
  message(FATAL_ERROR "Unsupported DTYPE_IN '${DTYPE_IN}' (bf16|i8|i16)")
endif()

if(DTYPE_OUT STREQUAL "bf16")
  set(_dtype_out_cpp test_utils::bfloat16_t)
  set(_dtype_acc_cpp float)
elseif(DTYPE_OUT STREQUAL "i16")
  set(_dtype_out_cpp int16_t)
  set(_dtype_acc_cpp int16_t)
elseif(DTYPE_OUT STREQUAL "i32")
  set(_dtype_out_cpp int32_t)
  set(_dtype_acc_cpp int32_t)
elseif(DTYPE_OUT STREQUAL "f32")
  set(_dtype_out_cpp float)
  set(_dtype_acc_cpp float)
elseif(DTYPE_OUT STREQUAL "i8")
  set(_dtype_out_cpp int8_t)
  set(_dtype_acc_cpp int8_t)
else()
  message(FATAL_ERROR "Unsupported DTYPE_OUT '${DTYPE_OUT}' (bf16|i8|i16|i32|f32)")
endif()

# matrix_vector/test.cpp hard-codes its own types (it #defines
# DATATYPES_USING_DEFINED before including ../test.cpp), so these are inert
# there -- same as under make, which passes them unconditionally too.
target_compile_definitions(${currentTarget} PRIVATE
    DTYPE_IN=${_dtype_in_cpp}
    DTYPE_OUT=${_dtype_out_cpp}
    DTYPE_ACC=${_dtype_acc_cpp}
)

# --- ${aieargs} --------------------------------------------------------------
# makefile-common's common tail, appended after each sibling's own aieargs.
set(_aieargs
    -M ${M} -K ${K} -N ${N}
    --dtype_in ${DTYPE_IN} --dtype_out ${DTYPE_OUT}
    --use-chess ${USE_CHESS}
    --emulate-bf16-mmul-with-bfp16 ${EMULATE_BF16_MMUL_WITH_BFP16})

# `run:` is `./$< -x .. -i .. -k MLIR_AIE -M $M -K $K -N $N ${runargs}`.
# Note that whole_array/ and single_core/ use `runargs+=` *before* including
# makefile-common, which turns the later `runargs?=-v 2 --warmup 1 --iters 1`
# into a no-op -- so those two siblings genuinely run without -v/--warmup/
# --iters. matrix_vector/ and cascade/ do get the defaults. Mirrored exactly.
set(_runargs -M ${M} -K ${K} -N ${N})

if(subdir STREQUAL "whole_array")
  list(PREPEND _aieargs -m ${TILE_M} -k ${TILE_K} -n ${TILE_N}
       --n-aie-cols ${N_AIE_COLS} --b-col-maj ${B_COL_MAJ} --c-col-maj ${C_COL_MAJ})
  if(USE_SCALAR)
    list(APPEND _aieargs --scalar 1)
  endif()
  list(APPEND _runargs --b_col_maj ${B_COL_MAJ} --c_col_maj ${C_COL_MAJ})
elseif(subdir STREQUAL "single_core")
  list(PREPEND _aieargs -m ${TILE_M} -k ${TILE_K} -n ${TILE_N}
       --b-col-maj ${B_COL_MAJ})
  list(APPEND _runargs --b_col_maj ${B_COL_MAJ})
elseif(subdir STREQUAL "cascade")
  list(PREPEND _aieargs -m ${TILE_M} -k ${TILE_K} -n ${TILE_N}
       --n-aie-cols ${N_AIE_COLS})
  list(APPEND _runargs -v 2 --warmup 1 --iters 1)
elseif(subdir STREQUAL "matrix_vector")
  # matrix_vector/Makefile adds no -m/-k to aieargs; matrix_vector.py's
  # defaults (m=32, k=32) are what the kernel is built for.
  if(USE_SCALAR)
    list(APPEND _aieargs --scalar)  # store_true here, unlike whole_array
  endif()
  list(APPEND _runargs -v 2 --warmup 1 --iters 1)
else()
  message(FATAL_ERROR
      "Unknown -Dsubdir='${subdir}' "
      "(single_core|whole_array|matrix_vector|cascade)")
endif()

add_aie_design(TARGET ${currentTarget} PY ${subdir}.py DEVICE ${AIE_DEVICE}
               DEVICE_FLAG --dev
               PY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}
               ARGS ${_aieargs})
add_dependencies(${currentTarget} ${currentTarget}_xclbin)
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} EXE ${currentTarget}
                 RUN_ARGS ${_runargs})

endif() # AIE_BUILD_DESIGN
