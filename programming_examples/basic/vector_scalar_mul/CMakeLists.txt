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

set(INT_BIT_WIDTH 16 CACHE STRING "integer bit width size")

# Buffer sizes default off INT_BIT_WIDTH, as the Makefile does: 8192 bytes for
# the 16-bit kernel, 16384 for the 32-bit one. This has to precede the IN1_SIZE /
# OUT_SIZE cache entries below, because the first set(... CACHE ...) wins -- and
# the same values feed both the design args and the host compile definitions, so
# a stale default here would silently mismatch the two.
if(INT_BIT_WIDTH EQUAL 16)
  set(_default_buf_size 8192)
else()
  set(_default_buf_size 16384)
endif()

set(IN1_SIZE ${_default_buf_size} CACHE STRING "in1 buffer size")
set(IN2_SIZE 4 CACHE STRING "in2 buffer size")
set(OUT_SIZE ${_default_buf_size} CACHE STRING "out buffer size")

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

add_executable(${currentTarget} test.cpp)

target_compile_definitions(${currentTarget} PUBLIC
    IN1_SIZE=${IN1_SIZE}
    IN2_SIZE=${IN2_SIZE}
    OUT_SIZE=${OUT_SIZE}
    INT_BIT_WIDTH=${INT_BIT_WIDTH}
)

target_include_directories (${currentTarget} PUBLIC ${XRT_INC_DIR})

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
)

target_link_test_utils(${currentTarget})


# Build the design and register the NPU run; -DAIE_DEVICE=npu2 targets Strix.
# The buffer sizes come from the INT_BIT_WIDTH block at the top of this file and
# feed the design args here as well as the host compile definitions above, so the
# two cannot drift. IN2_SIZE is always 4 (one int32 scalar factor).
# CHESS is not offered: xchesscc is unavailable on the native-Windows path this
# target exists to serve, so the design is always built with --use-chess 0.
set(AIE_DEVICE npu CACHE STRING "NPU device family for the example (npu|npu2)")
add_aie_design(TARGET ${currentTarget} PY vector_scalar_mul.py DEVICE ${AIE_DEVICE}
               ARGS -i1s ${IN1_SIZE} -i2s ${IN2_SIZE} -os ${OUT_SIZE}
                    -bw ${INT_BIT_WIDTH} --use-chess 0)
add_dependencies(${currentTarget} ${currentTarget}_xclbin)
add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} EXE ${currentTarget})
