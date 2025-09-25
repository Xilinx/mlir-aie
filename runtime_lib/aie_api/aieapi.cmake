
function(add_aie_api_headers SRCPATH BUILDPATH INSTALLPATH)
  message("Installing aie_api includes from ${SRCPATH} in ${INSTALLPATH}")

  # copy header files into install area
  install(DIRECTORY ${SRCPATH}/ DESTINATION ${INSTALLPATH}/aie_api)

  # copy header files from install area to build area for testing
  add_custom_command(
    OUTPUT ${BUILDPATH}/aie_api
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${INSTALLPATH}/aie_api ${BUILDPATH}/aie_api
    DEPENDS ${INSTALLPATH}/aie_api
    COMMENT "Copying aie_api includes from ${INSTALLPATH} to ${BUILDPATH}"
  )

endfunction()
