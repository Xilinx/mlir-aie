# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# third_party/aie-codegen (the successor to aie-rt) is consumed unpatched.
# Its sources live in src/ and its register-database headers in
# aie-regdb/globalparams.
set(AIE_CODEGEN_REGDB_SUBDIR "../aie-regdb/globalparams")

# aie-codegen's xaie_cdo.c includes the Vitis-only cdo_rts.h; provide it.
set(AIE_CODEGEN_SHIM_DIR "${CMAKE_CURRENT_LIST_DIR}/shim")

# Stages the driver headers in aie-codegen's native layout: aie_codegen.h at
# the root and everything else under aie_codegen_inc/. A Vitis aie-rt source
# dir has no aie_codegen.h, so it keeps its own layout and gets a forwarding
# aie_codegen.h, letting host code use a single include. Prebuilt LibXAIE
# installs are not staged here, so test_lib/target.h falls back to xaiengine.h.
function(add_aiert_headers TARGET SRCPATH BUILDPATH INSTALLPATH)
  message("Installing xaiengine includes for ${TARGET} from ${SRCPATH} in ${BUILDPATH}")
  cmake_path(ABSOLUTE_PATH SRCPATH BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
             NORMALIZE OUTPUT_VARIABLE _src)

  if(EXISTS ${_src}/aie_codegen.h)
    include(${_src}/cmake/AieCodegenHeaders.cmake)
    set(libheaders ${_src}/aie_codegen.h)
    set(libheadersSub)
    foreach(_dir ${_AIE_CODEGEN_HEADER_DIRS})
      file(GLOB _dirheaders ${_src}/${_dir}/*.h)
      list(APPEND libheadersSub ${_dirheaders})
    endforeach()
    file(GLOB _regdbheaders ${_src}/${AIE_CODEGEN_REGDB_SUBDIR}/*.h)
    list(APPEND libheadersSub ${_regdbheaders})
    set(_subdir aie_codegen_inc)
  else()
    file(GLOB libheaders ${_src}/*.h)
    file(GLOB libheadersSub ${_src}/*/*.h)
    set(_fwd ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}-compat/aie_codegen.h)
    file(WRITE ${_fwd} "#include <xaiengine.h>\n")
    list(APPEND libheaders ${_fwd})
    set(_subdir xaiengine)
  endif()

  # copy header files into build area
  set(_header_targets)
  foreach(file ${libheaders})
    cmake_path(GET file FILENAME basefile)
    set(dest ${BUILDPATH}/${basefile})
    add_custom_target(${TARGET}-${basefile} ALL DEPENDS ${dest})
    add_custom_command(
      OUTPUT ${dest}
      COMMAND ${CMAKE_COMMAND} -E copy ${file} ${dest}
      DEPENDS ${file})
    list(APPEND _header_targets ${TARGET}-${basefile})
  endforeach()

  foreach(file ${libheadersSub})
    cmake_path(GET file FILENAME basefile)
    set(dest ${BUILDPATH}/${_subdir}/${basefile})
    add_custom_target(${TARGET}-${basefile} ALL DEPENDS ${dest})
    add_custom_command(
      OUTPUT ${dest}
      COMMAND ${CMAKE_COMMAND} -E copy ${file} ${dest}
      DEPENDS ${file})
    list(APPEND _header_targets ${TARGET}-${basefile})
  endforeach()
  add_custom_target(${TARGET}-headers ALL DEPENDS ${_header_targets})

  # Install too
  install(FILES ${libheaders} DESTINATION ${INSTALLPATH})
  install(FILES ${libheadersSub} DESTINATION ${INSTALLPATH}/${_subdir})

endfunction()

function(add_aiert_library TARGET XAIE_SOURCE)
message("Building xaiengine library for ${TARGET} from ${SRCPATH}")
cmake_parse_arguments(ARG "STATIC" "" "" ${ARGN})
  if(ARG_STATIC)
    set(LIBTYPE STATIC)
  else()
    set(LIBTYPE SHARED)
  endif()

  file(GLOB libsources ${XAIE_SOURCE}/*/*.c ${XAIE_SOURCE}/*/*/*.c)
  set(_regdb ${XAIE_SOURCE}/${AIE_CODEGEN_REGDB_SUBDIR})

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
    ${XAIE_SOURCE}/util
    ${_regdb}
    ${AIE_CODEGEN_SHIM_DIR})

  add_library(${TARGET} ${LIBTYPE} ${libsources})
  set_property(TARGET ${TARGET} PROPERTY C_STANDARD 99)

  if (NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(${TARGET} PRIVATE -fPIC -Wno-gnu-designator)
  endif()

endfunction()
