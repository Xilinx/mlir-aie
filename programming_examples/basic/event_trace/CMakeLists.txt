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

# Set example-specific TARGET_NAME before mlir_aie_init_example() so its
# CACHE-form default (`test`) becomes a no-op.
set(TARGET_NAME mlir_trace_example CACHE STRING "Target to be built")

set(IN1_SIZE 16384 CACHE STRING "in1 buffer size")
set(IN2_SIZE 4 CACHE STRING "in2 buffer size")
set(OUT_SIZE 16384 CACHE STRING "out buffer size")

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

add_executable(${currentTarget}
    test.cpp
)

target_compile_definitions(${currentTarget} PUBLIC
    IN1_SIZE=${IN1_SIZE}
    IN2_SIZE=${IN2_SIZE}
    OUT_SIZE=${OUT_SIZE}
)

target_include_directories (${currentTarget} PUBLIC
    ${XRT_INC_DIR}
)

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
)

target_link_test_utils(${currentTarget})

# -----------------------------------------------------------------------------
# CMake-driven design build + NPU run (run_cmake.lit)
# -----------------------------------------------------------------------------
# The CMake equivalent of the Makefile's jit_xclbin plus the run halves of
# `run_trace` (test.cpp) and `run_trace_py` (test.py). The trace post-processing
# those recipes do afterwards -- parse.py + get_trace_summary.py -- lives in
# run_cmake.lit instead, so the ctests stay pure NPU runs.
#
# Each writes its own trace file rather than sharing the Makefile's trace.txt:
# both tests run in this one directory, so a shared name would leave only the
# last one's trace behind for the lit to parse.
if(AIE_BUILD_DESIGN)

set(TENSOR_SIZE 4096 CACHE STRING "Elements in the input tensor")
set(TILE_SIZE 1024 CACHE STRING "Elements per AIE tile iteration")
set(TRACE_SIZE 8192 CACHE STRING "Trace buffer size in bytes")

add_aie_design(TARGET ${currentTarget} PY aie_trace.py DEVICE ${AIE_DEVICE}
               ARGS --tensor-size ${TENSOR_SIZE} --tile-size ${TILE_SIZE}
                    --trace-size ${TRACE_SIZE})
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} EXE ${currentTarget}
                 RUN_ARGS -t ${TRACE_SIZE} --trace_file trace_cpp.txt)
add_aie_run_test(NAME ${currentTarget}_py DEVICE ${AIE_DEVICE} PY test.py
                 RUN_ARGS --verbosity 1 --trace_size ${TRACE_SIZE}
                          --trace-file trace_py.txt)

endif() # AIE_BUILD_DESIGN
