# Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# third_party/aie-codegen (the successor to aie-rt) is consumed unpatched.
# Its sources live in src/ and its register-database headers in
# aie-regdb/globalparams. Both are exposed to mlir-aie under xaiengine/.
set(AIE_CODEGEN_REGDB_SUBDIR "../aie-regdb/globalparams")

# aie-codegen's xaie_cdo.c includes the Vitis-only cdo_rts.h; provide it.
set(AIE_CODEGEN_SHIM_DIR "${CMAKE_CURRENT_LIST_DIR}/shim")

function(add_aiert_headers TARGET SRCPATH BUILDPATH INSTALLPATH)
  message("Installing aie-rt includes for ${TARGET} from ${SRCPATH} in ${BUILDPATH}")
  file(GLOB libheaders ${SRCPATH}/*.h)
  # aie-codegen's own umbrella includes <aie_codegen_inc/...>, a layout this
  # install does not create. mlir-aie exposes the xaiengine/ layout instead.
  list(FILTER libheaders EXCLUDE REGEX "/aie_codegen\\.h$")
  # aie-codegen has no xaiengine.h umbrella header (Vitis' aie-rt does).
  set(_has_umbrella FALSE)
  foreach(_h ${libheaders})
    cmake_path(GET _h FILENAME _name)
    if(_name STREQUAL "xaiengine.h")
      set(_has_umbrella TRUE)
    endif()
  endforeach()
  if(NOT _has_umbrella)
    list(APPEND libheaders ${AIE_CODEGEN_SHIM_DIR}/xaiengine.h)
  endif()
  file(GLOB libheadersSub ${SRCPATH}/*/*.h ${SRCPATH}/${AIE_CODEGEN_REGDB_SUBDIR}/*.h)

  # copy header files into build area
  set(_header_targets)
  foreach(file ${libheaders})
    cmake_path(GET file FILENAME basefile)
    # message("basefile: ${basefile}")
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
    # message("basefile: ${basefile}")
    set(dest ${BUILDPATH}/xaiengine/${basefile})
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
