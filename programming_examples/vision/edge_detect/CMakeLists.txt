# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# parameters
# -DOpenCV_DIR: Path to OpenCV install
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Set example-specific TARGET_NAME before mlir_aie_init_example() so its
# CACHE-form default (`test_exe`) becomes a no-op.
set(TARGET_NAME edge_detect CACHE STRING "Target to be built")

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()

if(NOT CMAKE_HOST_WIN32)
    set(OpenCV_DIR /usr/include/opencv4 CACHE STRING "Path to OpenCV install")
else()
    set(OpenCV_DIR C:/Technical/thirdParty/opencv/build CACHE STRING "Path to OpenCV install")
    add_compile_options("/std:c++17")
endif()

set(EDGEDETECT_WIDTH 1920 CACHE STRING "image width")
set(EDGEDETECT_HEIGHT 1080 CACHE STRING "image height")

project(${ProjectName})
include(../../common.cmake)

# Find packages
find_package(OpenCV REQUIRED)
message("opencv library path: ${OpenCV_LIB_PATH}")
message("opencv libs: ${OpenCV_LIBS}")

add_executable(${currentTarget}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/OpenCVUtils.cpp
    test.cpp
)

target_compile_definitions(${currentTarget} PUBLIC
        EDGEDETECT_WIDTH=${EDGEDETECT_WIDTH}
        EDGEDETECT_HEIGHT=${EDGEDETECT_HEIGHT}
)

target_include_directories (${currentTarget} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils
    ${XRT_INC_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

if(DEFINED XRT_INC_DIR AND XRT_INC_DIR)
    target_compile_definitions(${currentTarget} PRIVATE TEST_UTILS_USE_XRT)
endif()

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
    ${OpenCV_LIB_PATH}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
    ${OpenCV_LIBS}
)

target_link_test_utils(${currentTarget})

# -----------------------------------------------------------------------------
# CMake-driven design build + NPU run (run_cmake.lit)
# -----------------------------------------------------------------------------
# The CMake equivalent of the Makefile's jit_xclbin plus its `run` target.
# The `run_live` target is not converted: it opens a webcam.
#
# The design keeps the helper's default final.xclbin rather than the Makefile's
# build/final_<width>.xclbin: that name only ever existed so the `run` recipe
# could find it, and each configure has its own build directory here.
if(AIE_BUILD_DESIGN)

add_aie_design(TARGET ${currentTarget} PY edge_detect.py DEVICE ${AIE_DEVICE}
               ARGS -W ${EDGEDETECT_WIDTH} -H ${EDGEDETECT_HEIGHT})
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} EXE ${currentTarget})

endif() # AIE_BUILD_DESIGN
