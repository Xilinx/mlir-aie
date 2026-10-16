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

set(TARGET_NAME chaining_channels CACHE STRING "Target to be built")

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

add_executable(${currentTarget}
    test.cpp
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
# CMake-driven design build + NPU run (run_strix_cmake.lit)
# -----------------------------------------------------------------------------
# The CMake equivalent of the Makefile's jit_xclbin + `run:` recipe. Skipped
# when make drives the build (build_host_exe passes -DAIE_BUILD_DESIGN=OFF),
# which is what keeps the design from being JIT-ed twice.
if(AIE_BUILD_DESIGN)

# The design is aie2p-only, matching the single run_strix_makefile.lit.
set(AIE_DEVICE npu2 CACHE STRING "NPU device family for the example (npu2)")

set(LENGTH 1024 CACHE STRING "Vector length (Makefile `LENGTH`)")
set(TRACE 0 CACHE STRING "Trace buffer size in bytes, 0 to disable")

add_aie_design(TARGET ${currentTarget} PY chaining_channels.py DEVICE ${AIE_DEVICE}
               ARGS -n ${LENGTH} -t ${TRACE})
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} EXE ${currentTarget}
                 RUN_ARGS -l ${LENGTH} -t ${TRACE})

endif() # AIE_BUILD_DESIGN
