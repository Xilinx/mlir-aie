# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Fixture for helpers.lit: exercises every shape of add_aie_design() and
# add_aie_run_test() from common.cmake, including the misuse that must be
# rejected. Not an example -- there is no design and nothing runs on hardware.
#
# Two bugs shipped silently inside the CMake-native example path before this
# existed: enable_testing() called from function scope, which made every
# add_test() vanish while ctest still exited 0; and a testbench handed
# insts.bin when it loads its instruction stream with xrt::elf. Neither is
# visible without either hardware or an assertion on the generated build files.
# This fixture is the latter: configure-only, a few seconds, no NPU, and it
# behaves identically on Windows and Linux.
#
# Select a case with -DCASE=<name>; see helpers.lit for the full matrix.

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include(../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../common.cmake)

set(CASE all CACHE STRING "Helper behaviour to exercise (see helpers.lit)")

add_executable(host_exe main.cpp)

if(CASE STREQUAL "all")
  # One design, declared with ELF so it emits insts.bin *and* final.elf. That
  # lets the run tests below cover both instruction-stream choices without two
  # designs racing to write final.xclbin into the same directory.
  add_aie_design(TARGET host_exe PY design.py ELF ARGS --fixture-arg 7)

  # DEVICE omitted -> ${AIE_DEVICE}; the lit configures with -DAIE_DEVICE=npu2.
  add_aie_run_test(NAME t_exe EXE host_exe)
  add_aie_run_test(NAME t_exe_elf EXE host_exe USE_ELF)
  # Explicit DEVICE overrides the cache variable, and "npu" maps to run_on_npu.py's
  # "npu1" spelling.
  add_aie_run_test(NAME t_exe_npu1 DEVICE npu EXE host_exe)
  add_aie_run_test(NAME t_py PY host.py KERNEL MY_KERNEL
                   RUN_ARGS -l 4096 --op add
                   ENVIRONMENT "FIXTURE_VAR=fixture_value"
                   TIMEOUT 42)
  add_aie_run_test(NAME t_standalone PY design.py PY_STANDALONE RUN_ARGS --trace)

elseif(CASE STREQUAL "err_design_missing_py")
  add_aie_design(TARGET host_exe)

elseif(CASE STREQUAL "err_design_bad_py")
  add_aie_design(TARGET host_exe PY no_such_design.py)

elseif(CASE STREQUAL "err_design_bad_device")
  add_aie_design(TARGET host_exe PY design.py DEVICE npu3)

elseif(CASE STREQUAL "err_design_unknown_keyword")
  add_aie_design(TARGET host_exe PY design.py DEVICES npu)

elseif(CASE STREQUAL "err_design_missing_value")
  add_aie_design(TARGET host_exe PY design.py DEVICE)

elseif(CASE STREQUAL "err_run_missing_name")
  add_aie_design(TARGET host_exe PY design.py)
  add_aie_run_test(EXE host_exe)

elseif(CASE STREQUAL "err_run_no_host")
  add_aie_design(TARGET host_exe PY design.py)
  add_aie_run_test(NAME t)

elseif(CASE STREQUAL "err_run_two_hosts")
  add_aie_design(TARGET host_exe PY design.py)
  add_aie_run_test(NAME t EXE host_exe PY host.py)

elseif(CASE STREQUAL "err_run_exe_not_a_target")
  add_aie_design(TARGET host_exe PY design.py)
  add_aie_run_test(NAME t EXE no_such_target)

elseif(CASE STREQUAL "err_run_bad_py")
  add_aie_design(TARGET host_exe PY design.py)
  add_aie_run_test(NAME t PY no_such_host.py)

elseif(CASE STREQUAL "err_run_standalone_without_py")
  add_aie_design(TARGET host_exe PY design.py)
  add_aie_run_test(NAME t EXE host_exe PY_STANDALONE)

elseif(CASE STREQUAL "err_run_use_elf_without_elf_design")
  # The mismatch that made vector_scalar_add abort inside XRT: nothing in this
  # directory ever writes final.elf, so USE_ELF cannot be satisfied.
  add_aie_design(TARGET host_exe PY design.py)
  add_aie_run_test(NAME t EXE host_exe USE_ELF)

else()
  message(FATAL_ERROR "cmake_helpers: unknown CASE '${CASE}'")
endif()
