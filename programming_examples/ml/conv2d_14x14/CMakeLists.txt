# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# parameters
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built
# -DMULTI: Build the 32-core variant instead of the single-core one

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(TARGET_NAME conv2dk14 CACHE STRING "Target to be built")

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
# CMake-driven design build + NPU run (run_strix_cmake.lit)
# -----------------------------------------------------------------------------
# The CMake equivalent of the Makefile's jit_xclbin plus its `run_py` target.
# Both variants are aie2p only, so this stays npu2.
if(AIE_BUILD_DESIGN)

set(AIE_DEVICE npu2 CACHE STRING "NPU device family for the example (npu2)")

set(WIDTH 896 CACHE STRING "image width (Makefile `width`)")
set(HEIGHT 896 CACHE STRING "image height (Makefile `height`)")
option(MULTI "Build the 32-core (8 cols x 4 rows) variant" OFF)

# The single-core design sends the image co_group (= 1152/16 = 72) times in the
# input tensor; the multi-core design uses shim repeat and sends it once. This
# is the Makefile's `num_act` default.
if(MULTI)
  set(_multi --multi)
  set(_num_act 1)
else()
  set(_multi "")
  set(_num_act 72)
endif()

add_aie_design(TARGET ${currentTarget} PY conv2dk14.py DEVICE ${AIE_DEVICE}
               ARGS -wd ${WIDTH} -ht ${HEIGHT} ${_multi})
# -ic / -oc / -ksz are tied to the conv2dk14.cc kernel layer and are hardcoded
# in the Makefile's run_py recipe too; only -wd / -ht / -na follow MULTI.
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} PY test.py
                 RUN_ARGS -wd ${WIDTH} -ht ${HEIGHT}
                          -ic 4 -oc 1152 -ksz 14 -na ${_num_act})

endif() # AIE_BUILD_DESIGN
