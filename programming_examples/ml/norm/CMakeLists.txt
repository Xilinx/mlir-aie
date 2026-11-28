# Copyright (C) 2022 Xilinx, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# parameters
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Defaults match the Makefile's sequence_length/embedding_dim (and norm.py's
# own argparse defaults). The make flow always passed -DROWS/-DCOLS explicitly,
# so these were never exercised until the CMake run path started using them.
set(ROWS 64 CACHE STRING "row size of tensor (sequence length)")
set(COLS 4096 CACHE STRING "col size of tensor (embedding dim)")

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

add_executable(${currentTarget} test.cpp)

target_compile_definitions(${currentTarget} PUBLIC
    ROWS=${ROWS}
    COLS=${COLS}
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
# ROWS/COLS above are the sequence length / embedding dim: they feed both the
# design and the host's compile definitions, matching the Makefile's
# `-s ${sequence_length} -e ${embedding_dim}` + `-DROWS= -DCOLS=`.
# The op is selected through the NORM_OP env var, as in the Makefile's `run:`.
set(AIE_DEVICE npu CACHE STRING "NPU device family for the example (npu|npu2)")
set(NORM_OP rms CACHE STRING "Normalization op (rms|layer)")
add_aie_design(TARGET ${currentTarget} PY norm.py DEVICE ${AIE_DEVICE}
               ARGS -s ${ROWS} -e ${COLS} -o ${NORM_OP})
add_dependencies(${currentTarget} ${currentTarget}_xclbin)
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} EXE ${currentTarget}
                 ENVIRONMENT NORM_OP=${NORM_OP})
