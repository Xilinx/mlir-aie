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

set(TARGET_NAME custom_dma CACHE STRING "Target to be built")

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

# No add_executable() here: this example has no C++ host, only test.py. The
# project() call above still enables C/CXX, which is what lets common.cmake's
# find_package(XRT) import a shared library (see the note at its top).

# -----------------------------------------------------------------------------
# CMake-driven design build + NPU run (run_strix_cmake.lit)
# -----------------------------------------------------------------------------
# The CMake equivalent of the Makefile's build/aie.mlir + build/final.xclbin
# rules and its `run_py` target. custom_dma.py is not @iron.jit -- it prints
# MLIR on stdout and the Makefile drives aiecc itself -- so this is
# add_aie_mlir_design() rather than add_aie_design(). It needs no
# EMIT_MLIR_FLAG: printing is all the script does.
if(AIE_BUILD_DESIGN)

set(AIE_DEVICE npu2 CACHE STRING "NPU device family for the example (npu2)")

add_aie_mlir_design(TARGET ${currentTarget} PY custom_dma.py DEVICE ${AIE_DEVICE})
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} PY test.py)

endif() # AIE_BUILD_DESIGN
