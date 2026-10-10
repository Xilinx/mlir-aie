# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# parameters
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built
# -DGROUP: Which magika group to build (group0 | group2)
# -DTRACE_SIZE: Trace buffer size in bytes, 0 to disable

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(TARGET_NAME magika CACHE STRING "Target to be built")

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

# No add_executable() here: this example has no C++ host, only test_<group>.py.
# The project() call above still enables C/CXX, which is what lets
# common.cmake's find_package(XRT) import a shared library.

# -----------------------------------------------------------------------------
# CMake-driven design build + NPU run (run_cmake.lit and friends)
# -----------------------------------------------------------------------------
# The CMake equivalent of the Makefile's build/aie.mlir, its per-group kernel
# objects, build/final.xclbin and `run_py` / `trace_py`. The design scripts are
# not @iron.jit -- they print MLIR under --emit-mlir and the Makefile drives
# aiecc itself -- so this is add_aie_mlir_design().
#
# GROUP selects which of the Makefile's two `targetname` values is built, the
# way -Dsubdir does for the matmul sweeps. TRACE_SIZE > 0 selects the
# build/aie_trace.mlir + build/final_trace.xclbin half of the Makefile; the two
# are separate configures rather than two designs in one directory because
# upstream runs them from separate lits (and the group0 trace lit is XFAIL).
if(AIE_BUILD_DESIGN)

set(GROUP group0 CACHE STRING "Which magika group to build (group0|group2)")
set_property(CACHE GROUP PROPERTY STRINGS group0 group2)
if(NOT GROUP MATCHES "^group(0|2)$")
  message(FATAL_ERROR "magika: -DGROUP must be group0 or group2, got '${GROUP}'")
endif()

set(TRACE_SIZE 0 CACHE STRING "Trace buffer size in bytes, 0 to disable")

# The Makefile's aiecc lines pass ${AIECC_FLAGS} -v, and AIECC_FLAGS is empty
# unless AIE_CHECK_LUT_BANKS=1 (makefile-common:50). That opt-in is what adds
# --check-lut-banks, and it is also what makes the flag usable: it appends
# attach_bitcode to each kernel rule so the object keeps the LLVM IR the check
# reads. The flag alone fails here -- "'group0a.o' carries no readable LLVM IR".
# Mirror the default, which is the configuration upstream actually tests.
set(_aiecc_args -v)

# group0 links two objects built from one source under different -D, which is
# why these are spelled out per group rather than globbed. The object names are
# what the design scripts pass as `object_file_name`, and aiecc resolves them
# relative to its working directory -- so they must keep these exact names.
if(GROUP STREQUAL "group0")
  add_aie_kernel_object(OUTPUT group0a.o SOURCE kernels/group0.cc DEFINES GROUPA)
  add_aie_kernel_object(OUTPUT group0b.o SOURCE kernels/group0.cc DEFINES GROUPB)
  set(_objs group0a.o group0b.o)
else()
  add_aie_kernel_object(OUTPUT group2.o SOURCE kernels/group2.cc)
  set(_objs group2.o)
endif()

# test_<group>.py reads ./data/*.txt relative to its working directory, and the
# generated ctest runs in this build directory. The Makefile lits do the same
# thing with `cp -r %S/data .` before invoking make. (The design scripts read
# their LUTs relative to their own file, so they need no copy.)
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

if(TRACE_SIZE GREATER 0)
  # --get-input-with-addresses: python/utils/trace/parse.py reads
  # input_with_addresses.mlir to name the traced cores.
  add_aie_mlir_design(TARGET ${currentTarget} PY ${GROUP}.py DEVICE ${AIE_DEVICE}
                      EMIT_MLIR_FLAG --emit-mlir INPUT_WITH_ADDRESSES
                      ARGS -t ${TRACE_SIZE} AIECC_ARGS ${_aiecc_args}
                      OBJECTS ${_objs})
  add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} PY test_${GROUP}.py
                   RUN_ARGS --verbosity 1 --trace_size ${TRACE_SIZE})
else()
  add_aie_mlir_design(TARGET ${currentTarget} PY ${GROUP}.py DEVICE ${AIE_DEVICE}
                      EMIT_MLIR_FLAG --emit-mlir AIECC_ARGS ${_aiecc_args}
                      OBJECTS ${_objs})
  add_aie_run_test(NAME ${currentTarget} DEVICE ${AIE_DEVICE} PY test_${GROUP}.py
                   RUN_ARGS --verbosity 1)
endif()

endif() # AIE_BUILD_DESIGN
