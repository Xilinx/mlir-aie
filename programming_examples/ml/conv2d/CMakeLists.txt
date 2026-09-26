# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# parameters
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built
# -DFUSE_RELU: Fuse the ReLU into the conv kernel (Makefile `fuse_relu`)

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(TARGET_NAME conv2d CACHE STRING "Target to be built")

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
# FUSE_RELU is the Makefile's fuse_relu: it selects a different kernel library
# and has to be passed to both the design and the host test, which is why the
# Makefile threads it through aieargs and test_py_args alike.
if(AIE_BUILD_DESIGN)

set(WIDTH 32 CACHE STRING "conv tile width (Makefile `width`)")
set(HEIGHT 32 CACHE STRING "conv tile height (Makefile `height`)")
set(IN_CHANNELS 64 CACHE STRING "input channels (Makefile `in_channels`)")
set(OUT_CHANNELS 64 CACHE STRING "output channels (Makefile `out_channels`)")
option(FUSE_RELU "Fuse the ReLU into the conv kernel" OFF)

set(_relu "")
if(FUSE_RELU)
  set(_relu --fuse_relu)
endif()

add_aie_design(TARGET ${currentTarget} PY conv2d.py DEVICE ${AIE_DEVICE}
               ARGS -wd ${WIDTH} -ht ${HEIGHT}
                    -ic ${IN_CHANNELS} -oc ${OUT_CHANNELS} ${_relu})
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} PY test.py
                 RUN_ARGS -wd ${WIDTH} -ht ${HEIGHT}
                          -ic ${IN_CHANNELS} -oc ${OUT_CHANNELS} ${_relu})

endif() # AIE_BUILD_DESIGN
