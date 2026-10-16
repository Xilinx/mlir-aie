# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# parameters
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(TARGET_NAME row_wise_bias_add CACHE STRING "Target to be built")

set(SIZE_M 768 CACHE STRING "Rows (Makefile `M`)")
set(SIZE_N 2304 CACHE STRING "Columns (Makefile `N`)")
set(TILE_M 96 CACHE STRING "Tile rows (Makefile `m`)")
set(TILE_N 32 CACHE STRING "Tile columns (Makefile `n`)")

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

add_executable(${currentTarget}
    test.cpp
)

# test.cpp takes no argv: it #ifndef-guards XCLBIN / INSTS_BIN and bakes the
# paths in, so they have to point at this build directory rather than the
# Makefile's build/ defaults.
target_compile_definitions(${currentTarget} PUBLIC
    SIZE_M=${SIZE_M}
    SIZE_N=${SIZE_N}
    XCLBIN="${CMAKE_CURRENT_BINARY_DIR}/final.xclbin"
    INSTS_BIN="${CMAKE_CURRENT_BINARY_DIR}/insts.bin"
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
# The CMake equivalent of the Makefile's jit_xclbin + `run:` recipe. @iron.jit
# builds kernel.cc itself (via ExternalFunction.compile_flags), so there is no
# kernel object rule here either.
if(AIE_BUILD_DESIGN)

add_aie_design(TARGET ${currentTarget} PY row_wise_bias_add.py DEVICE ${AIE_DEVICE}
               ARGS -M ${SIZE_M} -N ${SIZE_N} -m ${TILE_M} -n ${TILE_N})
# `run:` is a bare `./build/test` -- NO_DEFAULT_ARGS keeps -x/-i/-k off the
# command line while still asserting the artifacts through REQUIRED_FILES.
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} EXE ${currentTarget}
                 NO_DEFAULT_ARGS)

endif() # AIE_BUILD_DESIGN
