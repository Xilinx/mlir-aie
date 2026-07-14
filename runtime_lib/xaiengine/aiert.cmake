# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# third_party/aie-rt tracks upstream Xilinx/aie-rt directly (no personal
# fork). Functionality mlir-aie needs that isn't upstream yet is vendored as
# patch files under third_party/patches/aie-rt and applied here, once, at
# configure time. See third_party/patches/aie-rt/README.md.
function(apply_aie_rt_vendor_patches AIE_RT_ROOT PATCH_DIR)
  # The vendored patches only modify existing upstream files, so detect an
  # already-patched tree by content rather than by a created-file sentinel:
  # 0001 replaces xaie_cdo.c's cdo_rts.h include with a cdo_Write32 forward
  # declaration that upstream does not have.
  set(_sentinel ${AIE_RT_ROOT}/driver/src/io_backend/ext/xaie_cdo.c)
  if(EXISTS ${_sentinel})
    file(READ ${_sentinel} _sentinel_contents)
    string(FIND "${_sentinel_contents}" "void cdo_Write32(" _sentinel_hit)
    if(NOT _sentinel_hit EQUAL -1)
      return()
    endif()
  endif()

  file(GLOB _patches ${PATCH_DIR}/*.patch)
  list(SORT _patches)
  find_package(Git REQUIRED)
  foreach(_patch ${_patches})
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
