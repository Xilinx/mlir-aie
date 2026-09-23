# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# third_party/aie-rt tracks upstream Xilinx/aie-rt directly (no personal
# fork). Functionality mlir-aie needs that isn't upstream yet is vendored as
# patch files under third_party/patches/aie-rt and applied here, once, at
# configure time. See third_party/patches/aie-rt/README.md.
function(apply_aie_rt_vendor_patches AIE_RT_ROOT PATCH_DIR)
  file(GLOB _patches ${PATCH_DIR}/*.patch)
  list(SORT _patches)
  find_package(Git REQUIRED)

  # A later patch may rewrite context lines of an earlier one, which then no
  # longer reverse-applies on its own. Replay the series onto HEAD in a scratch
  # index and find the longest prefix the patched files already match.
  set(_scratch_index ${CMAKE_CURRENT_BINARY_DIR}/aie-rt-patches.index)
  set(_git ${CMAKE_COMMAND} -E env GIT_INDEX_FILE=${_scratch_index}
    ${GIT_EXECUTABLE})
  set(_patched_paths)
  if(_patches)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} apply --numstat ${_patches}
      WORKING_DIRECTORY ${AIE_RT_ROOT}
      OUTPUT_VARIABLE _numstat
      ERROR_QUIET)
    string(REGEX MATCHALL "[^\n]+" _numstat_lines "${_numstat}")
    foreach(_line ${_numstat_lines})
      string(REGEX REPLACE "^[^\t]*\t[^\t]*\t" "" _path "${_line}")
      list(APPEND _patched_paths ${_path})
    endforeach()
    list(REMOVE_DUPLICATES _patched_paths)
  endif()
  execute_process(
    COMMAND ${_git} read-tree HEAD
    WORKING_DIRECTORY ${AIE_RT_ROOT}
    RESULT_VARIABLE _replay_result
    OUTPUT_QUIET ERROR_QUIET)
  set(_replayed 0)
  set(_applied_prefix 0)
  foreach(_patch ${_patches})
    if(NOT _replay_result EQUAL 0)
      break()
    endif()
    execute_process(
      COMMAND ${_git} apply --cached ${_patch}
      WORKING_DIRECTORY ${AIE_RT_ROOT}
      RESULT_VARIABLE _replay_result
      OUTPUT_QUIET ERROR_QUIET)
    if(_replay_result EQUAL 0)
      math(EXPR _replayed "${_replayed} + 1")
      execute_process(
        COMMAND ${_git} diff --quiet --no-ext-diff -- ${_patched_paths}
        WORKING_DIRECTORY ${AIE_RT_ROOT}
        RESULT_VARIABLE _tree_differs
        OUTPUT_QUIET ERROR_QUIET)
      if(_tree_differs EQUAL 0)
        set(_applied_prefix ${_replayed})
      endif()
    endif()
  endforeach()
  file(REMOVE ${_scratch_index})

  set(_position 0)
  foreach(_patch ${_patches})
    math(EXPR _position "${_position} + 1")
    if(_position LESS_EQUAL _applied_prefix)
      message(STATUS "Vendored aie-rt patch already applied, skipping: ${_patch}")
      continue()
    endif()
    # Past the matched prefix (or if the tree has other local edits), check
    # per patch: a clean reverse-apply check succeeds only when the tree
    # already contains it.
    execute_process(
      COMMAND ${GIT_EXECUTABLE} apply --reverse --check ${_patch}
      WORKING_DIRECTORY ${AIE_RT_ROOT}
      RESULT_VARIABLE _already_applied
      ERROR_QUIET)
    if(_already_applied EQUAL 0)
      message(STATUS "Vendored aie-rt patch already applied, skipping: ${_patch}")
      continue()
    endif()

    message(STATUS "Applying vendored aie-rt patch: ${_patch}")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} apply ${_patch}
      WORKING_DIRECTORY ${AIE_RT_ROOT}
      RESULT_VARIABLE _patch_result
      ERROR_VARIABLE _patch_error)
    if(NOT _patch_result EQUAL 0)
      message(FATAL_ERROR
        "Failed to apply vendored aie-rt patch ${_patch}:\n${_patch_error}\n"
        "If ${AIE_RT_ROOT} has local modifications, reset it with "
        "'git -C ${AIE_RT_ROOT} checkout -- .' and re-run CMake.")
    endif()
  endforeach()
endfunction()

function(add_aiert_headers TARGET SRCPATH BUILDPATH INSTALLPATH)
  message("Installing aie-rt includes for ${TARGET} from ${SRCPATH} in ${BUILDPATH}")
  file(GLOB libheaders ${SRCPATH}/*.h)
  file(GLOB libheadersSub ${SRCPATH}/*/*.h)

  # copy header files into build area
  foreach(file ${libheaders})
    cmake_path(GET file FILENAME basefile)
    # message("basefile: ${basefile}")
    set(dest ${BUILDPATH}/${basefile})
    add_custom_target(${TARGET}-${basefile} ALL DEPENDS ${dest})
    add_custom_command(
      OUTPUT ${dest}
      COMMAND ${CMAKE_COMMAND} -E copy ${file} ${dest}
      DEPENDS ${file})
  endforeach()

  set(_subheader_targets)
  foreach(file ${libheadersSub})
    cmake_path(GET file FILENAME basefile)
    # message("basefile: ${basefile}")
    set(dest ${BUILDPATH}/xaiengine/${basefile})
    add_custom_target(${TARGET}-${basefile} ALL DEPENDS ${dest})
    add_custom_command(
      OUTPUT ${dest}
      COMMAND ${CMAKE_COMMAND} -E copy ${file} ${dest}
      DEPENDS ${file})
    list(APPEND _subheader_targets ${TARGET}-${basefile})
  endforeach()
  add_custom_target(${TARGET}-headers ALL DEPENDS ${_subheader_targets})

  # Install too
  install(FILES ${libheaders} DESTINATION ${INSTALLPATH})
  install(FILES ${libheadersSub} DESTINATION ${INSTALLPATH}/xaiengine)

endfunction()

function(add_aiert_library TARGET XAIE_SOURCE)
message("Building aie-rt library for ${TARGET} from ${SRCPATH}")
cmake_parse_arguments(ARG "STATIC" "" "" ${ARGN})
  if(ARG_STATIC)
    set(LIBTYPE STATIC)
  else()
    set(LIBTYPE SHARED)
  endif()

  file(GLOB libsources ${XAIE_SOURCE}/*/*.c ${XAIE_SOURCE}/*/*/*.c)

  if(WIN32)
    list(FILTER libsources EXCLUDE REGEX xaie_sim\.c$)
  endif()

  include_directories(
    ${XAIE_SOURCE}
    ${XAIE_SOURCE}/common
    ${XAIE_SOURCE}/core
    ${XAIE_SOURCE}/device
    ${XAIE_SOURCE}/dma
    ${XAIE_SOURCE}/events
    ${XAIE_SOURCE}/global
    ${XAIE_SOURCE}/interrupt
    ${XAIE_SOURCE}/io_backend
    ${XAIE_SOURCE}/io_backend/ext
    ${XAIE_SOURCE}/io_backend/privilege
    ${XAIE_SOURCE}/lite
    ${XAIE_SOURCE}/locks
    ${XAIE_SOURCE}/memory
    ${XAIE_SOURCE}/npi
    ${XAIE_SOURCE}/perfcnt
    ${XAIE_SOURCE}/pl
    ${XAIE_SOURCE}/pm
    ${XAIE_SOURCE}/rsc
    ${XAIE_SOURCE}/stream_switch
    ${XAIE_SOURCE}/timer
    ${XAIE_SOURCE}/trace
    ${XAIE_SOURCE}/util)

  add_library(${TARGET} ${LIBTYPE} ${libsources})
  set_property(TARGET ${TARGET} PROPERTY C_STANDARD 99)

  if (NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(${TARGET} PRIVATE -fPIC -Wno-gnu-designator)
  endif()

endfunction()
