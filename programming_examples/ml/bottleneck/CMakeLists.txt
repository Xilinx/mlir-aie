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

set(TARGET_NAME bottleneck CACHE STRING "Target to be built")

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

# No add_executable() here: this example has no C++ host, only test.py. What
# this file held before was a copy of the vision template -- OpenCV, an
# EDGEDETECT_WIDTH/HEIGHT pair and a test.cpp that has never existed in this
# directory -- so it could not be configured at all. The project() call is
# still needed for common.cmake's find_package(XRT).

# -----------------------------------------------------------------------------
# CMake-driven design build + NPU run (run_cmake.lit)
# -----------------------------------------------------------------------------
# The CMake equivalent of the Makefile's jit_xclbin plus its `run_py` target.
if(AIE_BUILD_DESIGN)

add_aie_design(TARGET ${currentTarget} PY bottleneck.py DEVICE ${AIE_DEVICE})
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} PY test.py)

endif() # AIE_BUILD_DESIGN
