# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Pure-Python design: per_tile.py JITs the design and verifies it in-process,
# so there is no host test.cpp to build. add_aie_run_test's PY_STANDALONE mode
# runs the script on the NPU, mirroring the Makefile's `run_py` target.
# The script exits non-zero on mismatch (utils.verify.assert_pass), which is
# what the make lit's "CHECK: PASS!" was asserting.

cmake_minimum_required(VERSION 3.30)

include(../../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName} LANGUAGES NONE)
include(../../../common.cmake)

set(AIE_DEVICE npu CACHE STRING "NPU device family for the example (npu|npu2)")
set(TENSOR_HEIGHT 8 CACHE STRING "Tensor height")
set(TENSOR_WIDTH 8 CACHE STRING "Tensor width")
set(TILE_HEIGHT 2 CACHE STRING "Tile height")
set(TILE_WIDTH 2 CACHE STRING "Tile width")

add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} PY per_tile.py PY_STANDALONE
                 RUN_ARGS -d ${AIE_DEVICE}
                          --tensor-height ${TENSOR_HEIGHT}
                          --tensor-width ${TENSOR_WIDTH}
                          --tile-height ${TILE_HEIGHT}
                          --tile-width ${TILE_WIDTH})
