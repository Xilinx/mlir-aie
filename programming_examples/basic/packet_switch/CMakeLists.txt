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

set(TARGET_NAME packet_switch CACHE STRING "Target to be built")

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
# CMake-driven design build + NPU run (run_cmake.lit)
# -----------------------------------------------------------------------------
# Two designs from one packet_switch.py, one per --op, exactly as the Makefile's
# two jit_xclbin calls. OUTPUT_PREFIX is what keeps them from overwriting each
# other's final.xclbin; the artifact names match the Makefile's build/add.xclbin
# + build/add_insts.bin pairs.
if(AIE_BUILD_DESIGN)

add_aie_design(TARGET ${currentTarget}_add PY packet_switch.py DEVICE ${AIE_DEVICE}
               OUTPUT_PREFIX add ARGS --op add)
add_aie_design(TARGET ${currentTarget}_mult PY packet_switch.py DEVICE ${AIE_DEVICE}
               OUTPUT_PREFIX mult ARGS --op mul)
add_dependencies(${currentTarget} ${currentTarget}_add_xclbin ${currentTarget}_mult_xclbin)

# test.cpp reads <app_id> <insts> <xclbin> positionally, so the -x/-i/-k flags
# must not be synthesized. The artifacts are still asserted via REQUIRED_FILES.
add_aie_run_test(NAME ${currentTarget}_add DEVICE ${AIE_DEVICE} EXE ${currentTarget}
                 NO_DEFAULT_ARGS XCLBIN add.xclbin INSTS add_insts.bin
                 RUN_ARGS 0 "${CMAKE_CURRENT_BINARY_DIR}/add_insts.bin"
                          "${CMAKE_CURRENT_BINARY_DIR}/add.xclbin")
add_aie_run_test(NAME ${currentTarget}_mul DEVICE ${AIE_DEVICE} EXE ${currentTarget}
                 NO_DEFAULT_ARGS XCLBIN mult.xclbin INSTS mult_insts.bin
                 RUN_ARGS 1 "${CMAKE_CURRENT_BINARY_DIR}/mult_insts.bin"
                          "${CMAKE_CURRENT_BINARY_DIR}/mult.xclbin")

endif() # AIE_BUILD_DESIGN
