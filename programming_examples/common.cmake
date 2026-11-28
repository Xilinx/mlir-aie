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
# MSVC conformance flag
# -----------------------------------------------------------------------------
# Without /Zc:__cplusplus MSVC reports __cplusplus as 199711L regardless of
# /std:, which breaks headers that feature-test on it. It has to be added here
# rather than in mlir_aie_init_example(): that macro runs before project(), so
# MSVC is still undefined there, and adding the flag unconditionally breaks a
# native-Windows build whose compiler is a GNU-style clang (llvm-aie's, when it
# is on PATH) -- that driver rejects /Zc:__cplusplus as a missing input file.
if(MSVC)
  add_compile_options(/Zc:__cplusplus)
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
        # See mlir_aie_init.cmake for why this is CMAKE_HOST_WIN32 and not a
        # powershell.exe probe.
        if(NOT CMAKE_HOST_WIN32)
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
#
# A converted example needs exactly two calls after its add_executable():
#
#   add_aie_design(TARGET <exe> PY <design>.py [ELF] [ARGS ...])
#   add_aie_run_test(NAME <test> EXE <exe> [USE_ELF] ...)
#
# Both default DEVICE to the AIE_DEVICE cache variable below, and
# add_aie_design() wires the host exe's dependency on the JIT itself, so
# per-example boilerplate stays at those two lines.
#
# The helpers validate their arguments and FATAL_ERROR on misuse. That is
# deliberate: the failure modes here are quiet ones. A typo'd keyword, a design
# declared without ELF but run with USE_ELF, or a missing PY all used to
# configure cleanly and then either abort inside XRT or -- worse -- register no
# test at all, which ctest reports as success.

# Must be at directory scope: enable_testing() inside a function does NOT write
# CTestTestfile.cmake, so add_test() calls silently vanish and `ctest` reports
# "No tests were found!!!" -- and still exits 0, so the lit test passes without
# ever running on the NPU. This file is always included at directory scope.
enable_testing()

# Device family every example targets unless it overrides DEVICE explicitly.
# run_cmake.lit passes -DAIE_DEVICE=%aie_cmake_device%, resolved from the NPU lit
# actually detected on the machine.
set(AIE_DEVICE npu CACHE STRING "NPU device family for the examples (npu|npu2)")
set_property(CACHE AIE_DEVICE PROPERTY STRINGS npu npu2)

# Default wall-clock limit for a generated ctest, overridable per test with the
# TIMEOUT keyword. Without it a wedged NPU run is killed by lit's suite-wide
# timeout, which reports the whole lit test as timed out rather than naming the
# ctest that hung.
set(AIE_TEST_TIMEOUT 300 CACHE STRING "Default TIMEOUT (seconds) for generated NPU ctests")

# Reject unknown/typo'd keywords and keywords given without a value, then check
# that every argument in ARGN is set. Reads the caller's parsed variables
# directly -- CMake functions inherit the calling scope for reads.
function(_aie_validate_args _fn _prefix)
  if(DEFINED ${_prefix}_UNPARSED_ARGUMENTS)
    list(JOIN ${_prefix}_UNPARSED_ARGUMENTS " " _bad)
    message(FATAL_ERROR "${_fn}: unrecognized argument(s): ${_bad}")
  endif()
  if(DEFINED ${_prefix}_KEYWORDS_MISSING_VALUES)
    list(JOIN ${_prefix}_KEYWORDS_MISSING_VALUES " " _bad)
    message(FATAL_ERROR "${_fn}: keyword(s) given without a value: ${_bad}")
  endif()
  foreach(_arg IN LISTS ARGN)
    if(NOT ${_prefix}_${_arg})
      message(FATAL_ERROR "${_fn}: ${_arg} is required")
    endif()
  endforeach()
endfunction()

function(_aie_validate_device _fn _device)
  if(NOT _device MATCHES "^(npu|npu2)$")
    message(FATAL_ERROR
      "${_fn}: DEVICE must be 'npu' (Phoenix/Hawk) or 'npu2' (Strix), got '${_device}'")
  endif()
endfunction()

_aie_validate_device("-DAIE_DEVICE" "${AIE_DEVICE}")

# Required only by the helpers below, so host-only consumers don't need Python.
#
# The interpreter must be the one the IRON environment set up, because the design
# scripts import numpy and the `aie` package. Prefer an active virtualenv: on
# Windows CMake otherwise resolves Python from the registry and picks the system
# install, which has neither -- the JIT then dies with "No module named 'numpy'".
# (The main build sidesteps this by passing -DPython3_EXECUTABLE explicitly; the
# per-example configures get no such flag.) On POSIX this changes nothing, since
# FIRST is already CMake's default there and the venv is on PATH.
macro(_aie_require_python)
  if(NOT Python3_Interpreter_FOUND)
    set(Python3_FIND_VIRTUALENV FIRST)
    set(Python3_FIND_REGISTRY LAST)
    set(Python3_FIND_STRATEGY LOCATION)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
  endif()
endmacro()

# Artifact paths for a design. Without OUTPUT_PREFIX a design keeps
# final.xclbin / insts.bin / final.elf, which is what every existing example and
# every Makefile expects. A prefix is only needed when one directory builds more
# than one design, where the default names would collide -- the prefixed
# spelling mirrors what those Makefiles already do by hand (basic/packet_switch
# writes build/add.xclbin + build/add_insts.bin beside build/mult.xclbin).
function(_aie_design_artifacts _prefix _xclbin_var _insts_var _elf_var)
  if(_prefix)
    set(${_xclbin_var} "${CMAKE_CURRENT_BINARY_DIR}/${_prefix}.xclbin" PARENT_SCOPE)
    set(${_insts_var} "${CMAKE_CURRENT_BINARY_DIR}/${_prefix}_insts.bin" PARENT_SCOPE)
    set(${_elf_var} "${CMAKE_CURRENT_BINARY_DIR}/${_prefix}.elf" PARENT_SCOPE)
  else()
    set(${_xclbin_var} "${CMAKE_CURRENT_BINARY_DIR}/final.xclbin" PARENT_SCOPE)
    set(${_insts_var} "${CMAKE_CURRENT_BINARY_DIR}/insts.bin" PARENT_SCOPE)
    set(${_elf_var} "${CMAKE_CURRENT_BINARY_DIR}/final.elf" PARENT_SCOPE)
  endif()
endfunction()

# Record what a design emits, so add_aie_run_test() can resolve XCLBIN/INSTS by
# name and reject a name nothing produces. Called before the AIE_BUILD_DESIGN
# early return: under make the JIT happens outside CMake, but the artifacts the
# design describes are still the ones a test would consume.
function(_aie_register_design _target _xclbin _insts _elf _has_elf)
  set_property(DIRECTORY APPEND PROPERTY AIE_DESIGNS "${_target}")
  set_property(DIRECTORY APPEND PROPERTY AIE_DESIGN_ARTIFACTS "${_xclbin}" "${_insts}")
  if(_has_elf)
    set_property(DIRECTORY APPEND PROPERTY AIE_DESIGN_ARTIFACTS "${_elf}")
    set_property(DIRECTORY APPEND PROPERTY AIE_DESIGN_ELFS "${_elf}")
  endif()
endfunction()

# Resolve an XCLBIN/INSTS argument -- a bare artifact name or an absolute path --
# and require it to be something a design in this directory actually emits.
# A name that matches nothing is the quiet failure this guards: the test would
# still be registered, and the only symptom would be an abort inside XRT naming
# neither the test nor the typo.
function(_aie_resolve_artifact _fn _keyword _name _var)
  if(IS_ABSOLUTE "${_name}")
    set(_path "${_name}")
  else()
    set(_path "${CMAKE_CURRENT_BINARY_DIR}/${_name}")
  endif()
  get_directory_property(_known AIE_DESIGN_ARTIFACTS)
  if(NOT "${_path}" IN_LIST _known)
    list(JOIN _known "\n    " _listed)
    if(NOT _listed)
      set(_listed "(none -- this directory declares no design)")
    endif()
    # "Declared artifacts:" gets its own line: message() re-wraps running text
    # at a width we do not control, and it split that phrase across two lines.
    message(FATAL_ERROR
      "${_fn}: ${_keyword} '${_name}' is not emitted by any add_aie_design() or "
      "add_aie_mlir_design() in this directory."
      "\nDeclared artifacts:\n    ${_listed}")
  endif()
  set(${_var} "${_path}" PARENT_SCOPE)
endfunction()

# add_aie_design(TARGET <t> PY <design.py> [DEVICE <npu|npu2>] [ELF]
#                [OUTPUT_PREFIX <p>] [ARGS ...])
#   JITs the design into final.xclbin/insts.bin (+ final.elf with ELF) in the
#   build dir. Creates target <t>_xclbin, and makes <t> depend on it when <t> is
#   an existing target (pure-Python designs have no host exe, so that is
#   optional). DEVICE defaults to ${AIE_DEVICE}.
#
#   OUTPUT_PREFIX renames the artifacts to <p>.xclbin / <p>_insts.bin / <p>.elf.
#   Needed only when a directory builds more than one design, which would
#   otherwise have them overwrite each other; add_aie_run_test() then selects
#   between them with XCLBIN/INSTS.
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
  cmake_parse_arguments(D "ELF" "TARGET;PY;DEVICE;DEVICE_FLAG;PY_DIR;OUTPUT_PREFIX"
                          "ARGS" ${ARGN})
  _aie_validate_args("add_aie_design" D TARGET PY)
  if(NOT D_DEVICE)
    set(D_DEVICE "${AIE_DEVICE}")
  endif()
  _aie_validate_device("add_aie_design" "${D_DEVICE}")
  # Sweep families keep the design .py in a parent dir and drive it from a
  # per-parameterization subdir, so the script is not always beside the caller.
  # Resolved here rather than at first use so the existence check below sees it.
  set(_pydir "${CMAKE_CURRENT_SOURCE_DIR}")
  if(D_PY_DIR)
    set(_pydir "${D_PY_DIR}")
  endif()
  if(NOT EXISTS "${_pydir}/${D_PY}")
    message(FATAL_ERROR "add_aie_design: design script not found: ${_pydir}/${D_PY}")
  endif()

  # Record what this design emits, so add_aie_run_test can resolve XCLBIN/INSTS
  # by name and reject USE_ELF against a design that never emits one.
  _aie_design_artifacts("${D_OUTPUT_PREFIX}" _xclbin _insts _elf)
  _aie_register_design("${D_TARGET}" "${_xclbin}" "${_insts}" "${_elf}" "${D_ELF}")

  _aie_require_python()

  # Still define the target so callers' add_dependencies() stays valid; it just
  # has nothing to do.
  if(NOT AIE_BUILD_DESIGN)
    add_custom_target(${D_TARGET}_xclbin)
    if(TARGET ${D_TARGET})
      add_dependencies(${D_TARGET} ${D_TARGET}_xclbin)
    endif()
    return()
  endif()
  # Most designs take -d/--dev; the matmul family passes short_dev=None to
  # add_compile_args and so accepts only --dev.
  set(_devflag "-d")
  if(D_DEVICE_FLAG)
    set(_devflag "${D_DEVICE_FLAG}")
  endif()
  set(_out "${CMAKE_CURRENT_BINARY_DIR}")
  set(_outs ${_xclbin} ${_insts})
  set(_elfarg "")
  if(D_ELF)
    list(APPEND _outs "${_elf}")
    set(_elfarg "--elf-path=${_elf}")
  endif()
  add_custom_command(
    OUTPUT ${_outs}
    COMMAND ${Python3_EXECUTABLE} "${_pydir}/${D_PY}"
            ${_devflag} ${D_DEVICE} ${D_ARGS}
            "--xclbin-path=${_xclbin}" "--insts-path=${_insts}" ${_elfarg}
    DEPENDS "${_pydir}/${D_PY}"
    WORKING_DIRECTORY "${_out}"
    COMMENT "JIT-compiling ${D_PY} for ${D_DEVICE}"
    VERBATIM)
  add_custom_target(${D_TARGET}_xclbin ALL DEPENDS ${_outs})
  if(TARGET ${D_TARGET})
    add_dependencies(${D_TARGET} ${D_TARGET}_xclbin)
  endif()
endfunction()

# add_aie_run_test(NAME <t> [DEVICE <npu|npu2>] [EXE <host_target>] [PY <test.py>]
#                  [KERNEL <name>] [PY_STANDALONE] [USE_ELF] [NO_DEFAULT_ARGS]
#                  [XCLBIN <name>] [INSTS <name>] [TIMEOUT <secs>]
#                  [RUN_ARGS ...] [ENVIRONMENT ...])
#   Registers a ctest that runs on the NPU via utils/run_on_npu.py. Exactly one
#   of EXE or PY selects the host side:
#     EXE            => run the host binary against final.xclbin/insts.bin
#     PY             => run a Python host test against those artifacts (run_py)
#     PY_STANDALONE  => with PY, run the script alone (@iron.jit self-running
#                       designs) instead of passing it the built artifacts
#     USE_ELF        => pass final.elf instead of insts.bin as -i (xrt::elf +
#                       xrt::module testbenches; requires add_aie_design's ELF)
#     XCLBIN/INSTS   => select which design's artifacts, for a directory that
#                       builds several (see add_aie_design's OUTPUT_PREFIX).
#                       Each must name an artifact a design here declares.
#     NO_DEFAULT_ARGS => drop the -x/-i/-k flags and pass only RUN_ARGS. For
#                       host code that does not take them: basic/packet_switch
#                       reads <app_id> <insts> <xclbin> positionally, and
#                       basic/row_wise_bias_add reads no argv at all (its paths
#                       are compile definitions). The artifacts are still
#                       asserted via REQUIRED_FILES.
#     RUN_ARGS       => extra args appended to the host command, mirroring the
#                       Makefile `run:` recipe (e.g. -l 4096 --op add)
#     ENVIRONMENT    => "VAR=value" entries set for the test (e.g. NORM_OP=rms)
#     TIMEOUT        => seconds, default ${AIE_TEST_TIMEOUT}
#   DEVICE defaults to ${AIE_DEVICE}.
function(add_aie_run_test)
  cmake_parse_arguments(R "PY_STANDALONE;USE_ELF;NO_DEFAULT_ARGS"
                          "NAME;DEVICE;EXE;PY;KERNEL;TIMEOUT;XCLBIN;INSTS"
                          "RUN_ARGS;ENVIRONMENT" ${ARGN})
  _aie_validate_args("add_aie_run_test" R NAME)

  # Exactly one host side. Guarding this is what keeps a missing/empty PY from
  # silently falling through to the run_py branch and generating
  # `run_on_npu.py npu1 python <srcdir>/ --xclbin ...`, which fails far from
  # its cause.
  if(R_EXE AND R_PY)
    message(FATAL_ERROR "add_aie_run_test(${R_NAME}): EXE and PY are mutually exclusive")
  elseif(NOT R_EXE AND NOT R_PY)
    message(FATAL_ERROR "add_aie_run_test(${R_NAME}): one of EXE or PY is required")
  endif()
  if(R_EXE AND NOT TARGET ${R_EXE})
    message(FATAL_ERROR
      "add_aie_run_test(${R_NAME}): EXE '${R_EXE}' is not a target. "
      "Declare it with add_executable() before calling this.")
  endif()
  if(R_PY AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${R_PY}")
    message(FATAL_ERROR
      "add_aie_run_test(${R_NAME}): host script not found: ${CMAKE_CURRENT_SOURCE_DIR}/${R_PY}")
  endif()
  if(R_PY_STANDALONE AND NOT R_PY)
    message(FATAL_ERROR "add_aie_run_test(${R_NAME}): PY_STANDALONE requires PY")
  endif()

  # PY_STANDALONE runs an @iron.jit script that builds and loads its own design,
  # so it is handed no artifacts at all. Accepting keywords that only describe
  # artifacts would silently do nothing.
  if(R_PY_STANDALONE AND (R_XCLBIN OR R_INSTS OR R_USE_ELF OR R_NO_DEFAULT_ARGS))
    message(FATAL_ERROR
      "add_aie_run_test(${R_NAME}): PY_STANDALONE takes no artifact arguments, "
      "so XCLBIN/INSTS/USE_ELF/NO_DEFAULT_ARGS do not apply")
  endif()
  if(R_USE_ELF AND R_INSTS)
    message(FATAL_ERROR
      "add_aie_run_test(${R_NAME}): USE_ELF and INSTS both choose the "
      "instruction stream; pass INSTS <name>.elf alone instead")
  endif()
  # KERNEL only ever reaches the host as -k, which NO_DEFAULT_ARGS removes.
  if(R_NO_DEFAULT_ARGS AND R_KERNEL)
    message(FATAL_ERROR
      "add_aie_run_test(${R_NAME}): NO_DEFAULT_ARGS drops -k, so KERNEL has no effect")
  endif()

  if(NOT R_DEVICE)
    set(R_DEVICE "${AIE_DEVICE}")
  endif()
  _aie_validate_device("add_aie_run_test" "${R_DEVICE}")
  if(R_DEVICE STREQUAL "npu2")
    set(_kind npu2)
  else()
    set(_kind npu1)
  endif()
  set(_k MLIR_AIE)
  if(R_KERNEL)
    set(_k ${R_KERNEL})
  endif()

  # Which design's xclbin. Only a name given explicitly is checked against the
  # registry: the default has to stay usable by a directory whose artifacts are
  # produced outside CMake.
  if(R_XCLBIN)
    _aie_resolve_artifact("add_aie_run_test(${R_NAME})" XCLBIN "${R_XCLBIN}" _xclbin)
  else()
    set(_xclbin "${CMAKE_CURRENT_BINARY_DIR}/final.xclbin")
  endif()

  # The instruction stream is either the raw insts.bin or the ELF-wrapped form.
  # USE_ELF without a matching add_aie_design(... ELF) would hand the testbench
  # a file nothing ever writes; this is the reverse of the mismatch that made
  # vector_scalar_add abort inside XRT.
  if(R_INSTS)
    _aie_resolve_artifact("add_aie_run_test(${R_NAME})" INSTS "${R_INSTS}" _instr)
  elseif(R_USE_ELF)
    get_directory_property(_elfs AIE_DESIGN_ELFS)
    list(LENGTH _elfs _n_elfs)
    if(_n_elfs EQUAL 0)
      message(FATAL_ERROR
        "add_aie_run_test(${R_NAME}): USE_ELF requires a preceding "
        "add_aie_design(... ELF) in this directory to emit final.elf")
    elseif(_n_elfs GREATER 1)
      message(FATAL_ERROR
        "add_aie_run_test(${R_NAME}): this directory declares ${_n_elfs} ELF "
        "designs, so USE_ELF is ambiguous; pass INSTS <name>.elf instead")
    endif()
    list(GET _elfs 0 _instr)
  else()
    set(_instr "${CMAKE_CURRENT_BINARY_DIR}/insts.bin")
  endif()

  _aie_require_python()

  # Artifacts the test consumes, asserted via REQUIRED_FILES below so a missing
  # xclbin is reported by ctest instead of aborting inside XRT.
  set(_required "")
  # NO_DEFAULT_ARGS keeps the artifacts asserted but off the command line; the
  # host either takes them positionally or was compiled knowing their paths.
  if(R_NO_DEFAULT_ARGS)
    set(_default_exe_args "")
    set(_default_py_args "")
  else()
    set(_default_exe_args -x "${_xclbin}" -i "${_instr}" -k ${_k})
    set(_default_py_args --xclbin "${_xclbin}" --instr "${_instr}" -k ${_k})
  endif()

  if(R_EXE)
    set(_required "${_xclbin}" "${_instr}")
    add_test(NAME ${R_NAME}
      COMMAND ${Python3_EXECUTABLE} "${MLIR_AIE_DIR}/utils/run_on_npu.py" ${_kind}
              $<TARGET_FILE:${R_EXE}>
              ${_default_exe_args} ${R_RUN_ARGS})
  elseif(R_PY_STANDALONE)
    add_test(NAME ${R_NAME}
      COMMAND ${Python3_EXECUTABLE} "${MLIR_AIE_DIR}/utils/run_on_npu.py" ${_kind}
              ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/${R_PY}"
              ${R_RUN_ARGS})
  else()
    # `run_py` flow: a Python host test driven against the built artifacts.
    set(_required "${_xclbin}" "${_instr}")
    add_test(NAME ${R_NAME}
      COMMAND ${Python3_EXECUTABLE} "${MLIR_AIE_DIR}/utils/run_on_npu.py" ${_kind}
              ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/${R_PY}"
              ${_default_py_args} ${R_RUN_ARGS})
  endif()

  if(NOT R_TIMEOUT)
    set(R_TIMEOUT "${AIE_TEST_TIMEOUT}")
  endif()
  set_tests_properties(${R_NAME} PROPERTIES TIMEOUT ${R_TIMEOUT})
  if(_required)
    set_tests_properties(${R_NAME} PROPERTIES REQUIRED_FILES "${_required}")
  endif()
  if(R_ENVIRONMENT)
    set_tests_properties(${R_NAME} PROPERTIES ENVIRONMENT "${R_ENVIRONMENT}")
  endif()
endfunction()

# -----------------------------------------------------------------------------
# Explicit aiecc path (designs that are not @iron.jit)
# -----------------------------------------------------------------------------
# A handful of examples emit MLIR from their design script and then drive aiecc
# themselves, instead of letting @iron.jit do both. basic/custom_dma and
# ml/magika are the two. add_aie_mlir_design() is the CMake shape of that
# two-step recipe; everything downstream (add_aie_run_test, XCLBIN/INSTS
# selection, REQUIRED_FILES) works the same as for a JIT-ed design.

# aiecc ships beside the Python interpreter in a wheel install and in the build
# tree's bin/ from source, so look there before falling back to PATH. Resolved
# on demand rather than at include time: an example that never calls
# add_aie_mlir_design() must still configure without aiecc present.
macro(_aie_require_aiecc)
  if(NOT AIE_AIECC_EXECUTABLE)
    _aie_require_python()
    get_filename_component(_py_bin "${Python3_EXECUTABLE}" DIRECTORY)
    find_program(AIE_AIECC_EXECUTABLE NAMES aiecc
                 HINTS "${_py_bin}" "${MLIR_AIE_DIR}/bin" "${MLIR_AIE_DIR}/build/bin")
    if(NOT AIE_AIECC_EXECUTABLE)
      message(FATAL_ERROR
        "add_aie_mlir_design() needs 'aiecc', which was not found next to "
        "${Python3_EXECUTABLE}, under ${MLIR_AIE_DIR}/bin, or on PATH. "
        "Source utils/env_setup.sh, or set -DAIE_AIECC_EXECUTABLE=<path>.")
    endif()
  endif()
endmacro()

# add_aie_kernel_object(OUTPUT <name.o> SOURCE <kernel.cc> [DEVICE <npu|npu2>]
#                       [CHESS] [DEFINES ...] [INCLUDE_DIRS ...])
#   Compiles one AIE core function into ${CMAKE_CURRENT_BINARY_DIR}/<name.o>,
#   which is where aiecc resolves the `link_with` name from. Pass the same bare
#   <name.o> to add_aie_mlir_design's OBJECTS.
#
#   The compiler flags are not spelled out here. utils/compile_aie_kernel.py
#   forwards to compile_cxx_core_function(), which is the same code the JIT uses
#   and the single place the Peano/Chess flag sets live -- makefile-common's
#   PEANOWRAP2_FLAGS / CHESSCCWRAP2P_FLAGS are that list written out a second
#   time, and a third copy in CMake would be a third thing to keep in step.
function(add_aie_kernel_object)
  cmake_parse_arguments(K "CHESS" "OUTPUT;SOURCE;DEVICE" "DEFINES;INCLUDE_DIRS" ${ARGN})
  _aie_validate_args("add_aie_kernel_object" K OUTPUT SOURCE)
  if(NOT K_DEVICE)
    set(K_DEVICE "${AIE_DEVICE}")
  endif()
  _aie_validate_device("add_aie_kernel_object" "${K_DEVICE}")

  if(IS_ABSOLUTE "${K_SOURCE}")
    set(_src "${K_SOURCE}")
  else()
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${K_SOURCE}")
  endif()
  if(NOT EXISTS "${_src}")
    message(FATAL_ERROR "add_aie_kernel_object: kernel source not found: ${_src}")
  endif()

  _aie_require_python()

  set(_defargs "")
  foreach(_d IN LISTS K_DEFINES)
    list(APPEND _defargs -D "${_d}")
  endforeach()
  set(_incargs "")
  foreach(_i IN LISTS K_INCLUDE_DIRS)
    list(APPEND _incargs -I "${_i}")
  endforeach()
  set(_chessarg "")
  if(K_CHESS)
    set(_chessarg --chess)
  endif()

  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${K_OUTPUT}"
    COMMAND ${Python3_EXECUTABLE} "${MLIR_AIE_DIR}/utils/compile_aie_kernel.py"
            "${_src}" -o "${K_OUTPUT}" -d ${K_DEVICE}
            ${_defargs} ${_incargs} ${_chessarg}
    DEPENDS "${_src}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Compiling AIE kernel ${K_OUTPUT} for ${K_DEVICE}"
    VERBATIM)
endfunction()

# add_aie_mlir_design(TARGET <t> PY <design.py> [DEVICE <npu|npu2>]
#                     [OUTPUT_PREFIX <p>] [EMIT_MLIR_FLAG <flag>]
#                     [INPUT_WITH_ADDRESSES] [ARGS ...] [AIECC_ARGS ...]
#                     [OBJECTS ...])
#   Runs `python <design.py> -d <dev> [EMIT_MLIR_FLAG] [ARGS...] > <p>.mlir`,
#   then aiecc over that MLIR to produce the same artifacts add_aie_design()
#   does. OBJECTS names kernel objects (from add_aie_kernel_object) that the
#   design links, so aiecc reruns when a kernel changes.
#
#   EMIT_MLIR_FLAG is the switch the script needs to print MLIR instead of
#   building -- ml/magika takes --emit-mlir, basic/custom_dma prints by default.
#   INPUT_WITH_ADDRESSES additionally keeps input_with_addresses.mlir, which is
#   what python/utils/trace/parse.py reads to name the traced cores.
#
#   Guarded by AIE_BUILD_DESIGN for the same reason add_aie_design() is: under
#   make the design is built outside CMake, and doing it twice is wasted work.
function(add_aie_mlir_design)
  cmake_parse_arguments(M "INPUT_WITH_ADDRESSES"
                          "TARGET;PY;DEVICE;OUTPUT_PREFIX;EMIT_MLIR_FLAG"
                          "ARGS;AIECC_ARGS;OBJECTS" ${ARGN})
  _aie_validate_args("add_aie_mlir_design" M TARGET PY)
  if(NOT M_DEVICE)
    set(M_DEVICE "${AIE_DEVICE}")
  endif()
  _aie_validate_device("add_aie_mlir_design" "${M_DEVICE}")
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${M_PY}")
    message(FATAL_ERROR
      "add_aie_mlir_design: design script not found: ${CMAKE_CURRENT_SOURCE_DIR}/${M_PY}")
  endif()

  # This path never emits an ELF: aiecc's --get-npu-insts writes the raw stream.
  _aie_design_artifacts("${M_OUTPUT_PREFIX}" _xclbin _insts _elf)
  _aie_register_design("${M_TARGET}" "${_xclbin}" "${_insts}" "${_elf}" FALSE)

  _aie_require_python()

  if(NOT AIE_BUILD_DESIGN)
    add_custom_target(${M_TARGET}_xclbin)
    if(TARGET ${M_TARGET})
      add_dependencies(${M_TARGET} ${M_TARGET}_xclbin)
    endif()
    return()
  endif()

  _aie_require_aiecc()

  set(_stem "aie")
  if(M_OUTPUT_PREFIX)
    set(_stem "${M_OUTPUT_PREFIX}")
  endif()
  set(_mlir "${CMAKE_CURRENT_BINARY_DIR}/${_stem}.mlir")

  # The script prints MLIR on stdout, as the Makefile's `> $@` expects. VERBATIM
  # escapes a `>` into a literal argument, so the capture is done by the shim
  # rather than by giving up VERBATIM's quoting for the whole rule.
  add_custom_command(
    OUTPUT "${_mlir}"
    COMMAND ${Python3_EXECUTABLE} "${MLIR_AIE_DIR}/utils/emit_design_mlir.py"
            -o "${_mlir}" --
            ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/${M_PY}"
            -d ${M_DEVICE} ${M_EMIT_MLIR_FLAG} ${M_ARGS}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${M_PY}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Emitting MLIR from ${M_PY} for ${M_DEVICE}"
    VERBATIM)

  # aiecc takes artifact *names* relative to its working directory, and resolves
  # each kernel object the design links the same way, so it has to run in the
  # directory those objects were compiled into.
  get_filename_component(_xclbin_name "${_xclbin}" NAME)
  get_filename_component(_insts_name "${_insts}" NAME)
  set(_outs "${_xclbin}" "${_insts}")
  set(_iwa "")
  if(M_INPUT_WITH_ADDRESSES)
    set(_iwa --get-input-with-addresses)
    list(APPEND _outs "${CMAKE_CURRENT_BINARY_DIR}/input_with_addresses.mlir")
  endif()

  set(_objs "")
  foreach(_o IN LISTS M_OBJECTS)
    list(APPEND _objs "${CMAKE_CURRENT_BINARY_DIR}/${_o}")
  endforeach()

  add_custom_command(
    OUTPUT ${_outs}
    COMMAND "${AIE_AIECC_EXECUTABLE}" ${M_AIECC_ARGS}
            --get-xclbin "--xclbin-name=${_xclbin_name}"
            --get-npu-insts "--npu-insts-name=${_insts_name}"
            ${_iwa} "${_mlir}"
    DEPENDS "${_mlir}" ${_objs}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Compiling ${_stem}.mlir with aiecc for ${M_DEVICE}"
    VERBATIM)

  add_custom_target(${M_TARGET}_xclbin ALL DEPENDS ${_outs})
  if(TARGET ${M_TARGET})
    add_dependencies(${M_TARGET} ${M_TARGET}_xclbin)
  endif()
endfunction()
