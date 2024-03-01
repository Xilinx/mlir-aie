
function(add_aiert_headers TARGET SRCPATH BUILDPATH INSTALLPATH)
  message("${TARGET} ${SRCPATH} ${BUILDPATH}")
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
  cmake_parse_arguments(ARG "STATIC" "" "" ${ARGN})
  if(ARG_STATIC)
    set(LIBTYPE STATIC)
  else()
    set(LIBTYPE SHARED)
  endif()

  file(GLOB libsources ${XAIE_SOURCE}/*/*.c ${XAIE_SOURCE}/*/*/*.c)

  if(WIN32)
    list(FILTER libsources EXCLUDE REGEX xaie_amdair)
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
