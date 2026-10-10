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

set(PASSTHROUGH_SIZE 4096 CACHE STRING "size")

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

add_executable(${currentTarget} test.cpp)

target_compile_definitions(${currentTarget} PUBLIC
        PASSTHROUGH_SIZE=${PASSTHROUGH_SIZE}
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

# -----------------------------------------------------------------------------
# CMake-driven design build + NPU run (run_cmake.lit)
# -----------------------------------------------------------------------------
# The CMake equivalent of the Makefile's jit_xclbin plus both of its run
# recipes, `run` (test.cpp) and `run_py` (test.py). Skipped when make drives the
# build -- its ${targetname}_${data_size}.exe rule passes -DAIE_BUILD_DESIGN=OFF
# so the design is not JIT-ed a second time.
if(AIE_BUILD_DESIGN)

add_aie_design(TARGET ${currentTarget} PY passthrough_pykernel.py DEVICE ${AIE_DEVICE}
               ARGS -s ${PASSTHROUGH_SIZE})
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} EXE ${currentTarget})
add_aie_run_test(NAME ${currentTarget}_py DEVICE ${AIE_DEVICE} PY test.py
                 RUN_ARGS -s ${PASSTHROUGH_SIZE})

endif() # AIE_BUILD_DESIGN
