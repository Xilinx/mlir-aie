# Copyright (C) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Common CMake configuration for programming examples
# This file provides common setup for test_utils library linking

# -----------------------------------------------------------------------------
# Guard: project() must be called before this file is included, because
# find_package(XRT) below loads xrt-targets.cmake which calls
# add_library(... SHARED IMPORTED). Without a prior project() call, CMake
# has not initialised platform shared-library support and the call either
# fails ("does not support dynamic linking") or silently downgrades to
# STATIC. See Xilinx/mlir-aie#3048.
# -----------------------------------------------------------------------------
if(NOT PROJECT_NAME)
  message(FATAL_ERROR
    "common.cmake must be included after project(). "
    "Call mlir_aie_init_example() (or your own project() call) first. "
    "See https://github.com/Xilinx/mlir-aie/issues/3048")
endif()

# -----------------------------------------------------------------------------
# Resolve MLIR-AIE root directory
# -----------------------------------------------------------------------------
# In WSL, CMake runs on Windows via `powershell.exe cmake`. Therefore, we must
# prefer deterministic repo-root detection. Fall back to Python only if needed.

get_filename_component(_mlir_aie_repo_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(EXISTS "${_mlir_aie_repo_root}/runtime_lib/test_lib/xrt_test_wrapper.h")
  set(MLIR_AIE_DIR "${_mlir_aie_repo_root}")
else()
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "from aie.utils.config import root_path; print(root_path())"
      OUTPUT_VARIABLE MLIR_AIE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
  endif()
endif()

if(NOT MLIR_AIE_DIR)
  message(FATAL_ERROR "Unable to determine MLIR_AIE_DIR (repo root not found and Python probe unavailable).")
endif()

# Make the repo's Find modules (FindHRX.cmake, ...) available to find_package.
list(APPEND CMAKE_MODULE_PATH "${MLIR_AIE_DIR}/cmake/modules")

# -----------------------------------------------------------------------------
# HRX backend selection (RUNTIME=hrx in makefile-common -> -DUSE_HRX=ON)
# -----------------------------------------------------------------------------
# When building the HRX host backend, examples dispatch through libhrx instead
# of XRT. We don't need the XRT SDK headers at all, but the per-example
# CMakeLists still does `target_link_libraries(... xrt_coreutil)` and
# `target_include_directories(... ${XRT_INC_DIR})`. To keep those a no-op
# without editing ~50 example files, define a dummy INTERFACE target named
# `xrt_coreutil` (so the link resolves to nothing instead of `-lxrt_coreutil`)
# and leave the XRT include/lib dir variables empty.
option(USE_HRX "Build programming-example host code against HRX instead of XRT" OFF)

if(USE_HRX)
  if(NOT TARGET xrt_coreutil)
    add_library(xrt_coreutil INTERFACE IMPORTED)
  endif()
  if(NOT DEFINED XRT_INC_DIR)
    set(XRT_INC_DIR "" CACHE STRING "Path to XRT headers (unused for HRX)")
  endif()
  if(NOT DEFINED XRT_LIB_DIR)
    set(XRT_LIB_DIR "" CACHE STRING "Path to XRT libraries (unused for HRX)")
  endif()
endif()

# -----------------------------------------------------------------------------
# XRT auto-detection (supports both Ubuntu packages and legacy /opt/xilinx/xrt)
# -----------------------------------------------------------------------------
if(NOT USE_HRX)
if(NOT DEFINED XRT_INC_DIR OR NOT DEFINED XRT_LIB_DIR)
    find_package(XRT QUIET)
    if(XRT_FOUND)
        # find_package(XRT) may resolve via the project's FindXRT.cmake (which
        # sets XRT_INCLUDE_DIR / XRT_LIB_DIR, singular) or via XRT's own
        # xrt-config.cmake (which sets XRT_INCLUDE_DIRS / XRT_LINK_DIRS,
        # plural).  Accept whichever set is available.
        if(NOT DEFINED XRT_INC_DIR)
            if(XRT_INCLUDE_DIRS)
                set(XRT_INC_DIR "${XRT_INCLUDE_DIRS}" CACHE STRING "Path to XRT headers")
            elseif(XRT_INCLUDE_DIR)
                set(XRT_INC_DIR "${XRT_INCLUDE_DIR}" CACHE STRING "Path to XRT headers")
            endif()
        endif()
        if(NOT DEFINED XRT_LIB_DIR)
            if(XRT_LINK_DIRS)
                set(XRT_LIB_DIR "${XRT_LINK_DIRS}" CACHE STRING "Path to XRT libraries")
            endif()
        endif()
    endif()

    # Fall back to legacy/default paths if still unset
    if(NOT DEFINED XRT_INC_DIR OR NOT DEFINED XRT_LIB_DIR)
        find_program(WSL NAMES powershell.exe)
        if(NOT WSL)
            if(NOT DEFINED XRT_INC_DIR)
                set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT headers")
            endif()
            if(NOT DEFINED XRT_LIB_DIR)
                set(XRT_LIB_DIR /opt/xilinx/xrt/lib CACHE STRING "Path to XRT libraries")
            endif()
        else()
            if(NOT DEFINED XRT_INC_DIR)
                set(XRT_INC_DIR C:/Technical/XRT/src/runtime_src/core/include CACHE STRING "Path to XRT headers")
            endif()
            if(NOT DEFINED XRT_LIB_DIR)
                set(XRT_LIB_DIR C:/Technical/xrtNPUfromDLL CACHE STRING "Path to XRT libraries")
            endif()
        endif()
    endif()
endif()
endif() # NOT USE_HRX

# -----------------------------------------------------------------------------
# test_utils discovery
# -----------------------------------------------------------------------------
# Preferred: installed layout (from cmake --install). Fallback: build from source.
set(TEST_UTILS_INST_LIB_DIR "${MLIR_AIE_DIR}/runtime_lib/x86_64/test_lib/lib")
set(TEST_UTILS_INST_INC_DIR "${MLIR_AIE_DIR}/runtime_lib/x86_64/test_lib/include")
set(TEST_UTILS_SRC_DIR     "${MLIR_AIE_DIR}/runtime_lib/test_lib")
set(TEST_UTILS_RUNTIME_LIB_DIR "${MLIR_AIE_DIR}/runtime_lib")

function(target_link_test_utils target_name)
  target_include_directories(${target_name} PUBLIC "${TEST_UTILS_RUNTIME_LIB_DIR}")

  # 0) HRX backend: dispatch via libhrx, no XRT SDK needed. test_utils is built
  #    WITHOUT TEST_UTILS_USE_XRT (its XRT block is #ifdef'd out and unused by
  #    the HRX wrapper), and the example target gets TEST_UTILS_USE_HRX so
  #    xrt_test_wrapper.h pulls in hrx_test_wrapper.h.
  if(USE_HRX)
    if(NOT EXISTS "${TEST_UTILS_SRC_DIR}/hrx_test_wrapper.h")
      message(FATAL_ERROR "HRX wrapper not found at: ${TEST_UTILS_SRC_DIR}")
    endif()

    # Auto-detect HRX (FindHRX.cmake probes standard locations + env hints and
    # prefers the shipped hrx CMake package). Done once at function scope; HRX_*
    # persist as cache vars afterwards. libhrx now builds the amdxdna XADX
    # package internally, so the `runtime` component (headers + libhrx) is all
    # an example needs to link.
    if(NOT HRX_FOUND)
      find_package(HRX QUIET COMPONENTS runtime)
    endif()
    if(NOT HRX_FOUND)
      message(FATAL_ERROR
        "USE_HRX=ON but the HRX runtime was not found. "
        "Set HRX_DIR (source checkout with libhrx/include/hrx_runtime.h) and "
        "LIBHRX_DIR (dir with libhrx.so), or install HRX to a standard "
        "location. Falling back to the default XRT backend (RUNTIME=xrt) is "
        "also an option if HRX is unavailable.")
    endif()

    target_include_directories(${target_name} PUBLIC
        "${TEST_UTILS_SRC_DIR}" "${HRX_INCLUDE_DIR}")
    target_compile_definitions(${target_name} PRIVATE TEST_UTILS_USE_HRX)

    if(NOT TARGET test_utils)
      add_library(test_utils STATIC "${TEST_UTILS_SRC_DIR}/test_utils.cpp")
      target_include_directories(test_utils PUBLIC
          "${TEST_UTILS_SRC_DIR}" "${TEST_UTILS_RUNTIME_LIB_DIR}")
    endif()

    target_link_libraries(${target_name} PUBLIC test_utils "${HRX_LIBHRX}")
    return()
  endif()

  # 1) Use installed/prebuilt if present
  if(EXISTS "${TEST_UTILS_INST_INC_DIR}/xrt_test_wrapper.h" AND EXISTS "${TEST_UTILS_INST_LIB_DIR}")
    target_include_directories(${target_name} PUBLIC "${TEST_UTILS_INST_INC_DIR}")
    target_link_directories(${target_name} PUBLIC "${TEST_UTILS_INST_LIB_DIR}")
    target_link_libraries(${target_name} PUBLIC test_utils)
    return()
  endif()

  # 2) Otherwise build test_utils from source
  if(NOT EXISTS "${TEST_UTILS_SRC_DIR}/test_utils.cpp")
    message(FATAL_ERROR "test_utils source not found at: ${TEST_UTILS_SRC_DIR}")
  endif()

  target_include_directories(${target_name} PUBLIC "${TEST_UTILS_SRC_DIR}")

  if(NOT TARGET test_utils)
    add_library(test_utils STATIC "${TEST_UTILS_SRC_DIR}/test_utils.cpp")
    target_include_directories(test_utils PUBLIC "${TEST_UTILS_SRC_DIR}" "${TEST_UTILS_RUNTIME_LIB_DIR}")

    # Enable XRT helpers if an XRT include dir is available
    if(DEFINED XRT_INC_DIR AND XRT_INC_DIR)
      target_include_directories(test_utils PUBLIC "${XRT_INC_DIR}")
      target_compile_definitions(test_utils PRIVATE TEST_UTILS_USE_XRT)
    elseif(DEFINED XRT_INCLUDE_DIRS AND XRT_INCLUDE_DIRS)
      target_include_directories(test_utils PUBLIC "${XRT_INCLUDE_DIRS}")
      target_compile_definitions(test_utils PRIVATE TEST_UTILS_USE_XRT)
    elseif(DEFINED XRT_INCLUDE_DIR AND XRT_INCLUDE_DIR)
      target_include_directories(test_utils PUBLIC "${XRT_INCLUDE_DIR}")
      target_compile_definitions(test_utils PRIVATE TEST_UTILS_USE_XRT)
    endif()
  endif()

  target_link_libraries(${target_name} PUBLIC test_utils)
endfunction()

# -----------------------------------------------------------------------------
# Make-free NPU design build + run helpers
# -----------------------------------------------------------------------------
# CMake equivalents of makefile-common's jit_xclbin and the per-example `run:`
# target: build the xclbin/insts and run on the NPU via cmake + ctest.

# Must be at directory scope: enable_testing() inside a function does NOT write
# CTestTestfile.cmake, so add_test() calls silently vanish and `ctest` reports
# "No tests were found!!!" -- and still exits 0, so the lit test passes without
# ever running on the NPU. This file is always included at directory scope.
enable_testing()

# Required only by the helpers below, so host-only consumers don't need Python.
macro(_aie_require_python)
  if(NOT Python3_Interpreter_FOUND)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
  endif()
endmacro()

# add_aie_design(TARGET <t> PY <design.py> DEVICE <npu|npu2> [ELF] [ARGS ...])
#   JITs the design into final.xclbin/insts.bin (+ final.elf with ELF) in the
#   build dir. Creates target <t>_xclbin for the host exe to depend on.
#
# The design is only built when AIE_BUILD_DESIGN is ON. This matters because
# makefile-common's build_host_exe configures and builds this same CMakeLists to
# produce the host binary, while separately JIT-ing the design itself via
# jit_xclbin. Without the guard, `make` would also trigger the CMake-side JIT --
# duplicated work that additionally broke ml/block_datatypes (BFP kernels fail to
# compile in that context). build_host_exe therefore passes -DAIE_BUILD_DESIGN=OFF,
# and run_cmake.lit gets the default ON.
option(AIE_BUILD_DESIGN "Build the example's AIE design (off when make drives the build)" ON)

function(add_aie_design)
  _aie_require_python()
  cmake_parse_arguments(D "ELF" "TARGET;PY;DEVICE" "ARGS" ${ARGN})
  # Still define the target so callers' add_dependencies() stays valid; it just
  # has nothing to do.
  if(NOT AIE_BUILD_DESIGN)
    add_custom_target(${D_TARGET}_xclbin)
    return()
  endif()
  set(_out "${CMAKE_CURRENT_BINARY_DIR}")
  set(_xclbin "${_out}/final.xclbin")
  set(_insts "${_out}/insts.bin")
  set(_outs ${_xclbin} ${_insts})
  set(_elfarg "")
  if(D_ELF)
    list(APPEND _outs "${_out}/final.elf")
    set(_elfarg "--elf-path=${_out}/final.elf")
  endif()
  add_custom_command(
    OUTPUT ${_outs}
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/${D_PY}"
            -d ${D_DEVICE} ${D_ARGS}
            "--xclbin-path=${_xclbin}" "--insts-path=${_insts}" ${_elfarg}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${D_PY}"
    WORKING_DIRECTORY "${_out}"
    COMMENT "JIT-compiling ${D_PY} for ${D_DEVICE}"
    VERBATIM)
  add_custom_target(${D_TARGET}_xclbin ALL DEPENDS ${_outs})
endfunction()

# add_aie_run_test(NAME <t> DEVICE <npu|npu2> [EXE <host_target>] [PY <test.py>]
#                  [KERNEL <name>] [PY_STANDALONE] [USE_ELF]
#                  [RUN_ARGS ...] [ENVIRONMENT ...])
#   Registers a ctest that runs on the NPU via utils/run_on_npu.py.
#     EXE            => run the host binary against final.xclbin/insts.bin
#     PY             => run a Python host test against those artifacts (run_py)
#     PY_STANDALONE  => run the script alone (@iron.jit self-running designs)
#     USE_ELF        => pass final.elf instead of insts.bin as -i (xrt::elf +
#                       xrt::module testbenches; pair with add_aie_design's ELF)
#     RUN_ARGS       => extra args appended to the host command, mirroring the
#                       Makefile `run:` recipe (e.g. -l 4096 --op add)
#     ENVIRONMENT    => "VAR=value" entries set for the test (e.g. NORM_OP=rms)
function(add_aie_run_test)
  _aie_require_python()
  cmake_parse_arguments(R "PY_STANDALONE;USE_ELF" "NAME;DEVICE;EXE;PY;KERNEL"
                          "RUN_ARGS;ENVIRONMENT" ${ARGN})
  if(R_DEVICE STREQUAL "npu2")
    set(_kind npu2)
  else()
    set(_kind npu1)
  endif()
  set(_k MLIR_AIE)
  if(R_KERNEL)
    set(_k ${R_KERNEL})
  endif()
  # The instruction stream is either the raw insts.bin or the ELF-wrapped form.
  if(R_USE_ELF)
    set(_instr "${CMAKE_CURRENT_BINARY_DIR}/final.elf")
  else()
    set(_instr "${CMAKE_CURRENT_BINARY_DIR}/insts.bin")
  endif()
  if(R_EXE)
    add_test(NAME ${R_NAME}
      COMMAND ${Python3_EXECUTABLE} "${MLIR_AIE_DIR}/utils/run_on_npu.py" ${_kind}
              $<TARGET_FILE:${R_EXE}>
              -x "${CMAKE_CURRENT_BINARY_DIR}/final.xclbin"
              -i "${_instr}"
              -k ${_k} ${R_RUN_ARGS})
  elseif(R_PY_STANDALONE)
    add_test(NAME ${R_NAME}
      COMMAND ${Python3_EXECUTABLE} "${MLIR_AIE_DIR}/utils/run_on_npu.py" ${_kind}
              ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/${R_PY}"
              ${R_RUN_ARGS})
  else()
    # `run_py` flow: a Python host test driven against the built artifacts.
    add_test(NAME ${R_NAME}
      COMMAND ${Python3_EXECUTABLE} "${MLIR_AIE_DIR}/utils/run_on_npu.py" ${_kind}
              ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/${R_PY}"
              --xclbin "${CMAKE_CURRENT_BINARY_DIR}/final.xclbin"
              --instr "${_instr}"
              -k ${_k} ${R_RUN_ARGS})
  endif()
  if(R_ENVIRONMENT)
    set_tests_properties(${R_NAME} PROPERTIES ENVIRONMENT "${R_ENVIRONMENT}")
  endif()
endfunction()
