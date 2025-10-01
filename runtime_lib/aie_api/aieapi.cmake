
function(add_aie_api_headers SRCPATH BUILDPATH INSTALLPATH)
  message("Installing aie_api includes from ${SRCPATH} in ${INSTALLPATH}")

  # copy header files into install area
  install(DIRECTORY ${SRCPATH}/ DESTINATION ${INSTALLPATH}/aie_api)

  message("Copying aie_api includes from ${SRCPATH} to ${BUILDPATH}")

  # copy header files from install area to build area for testing
  file(COPY ${SRCPATH} DESTINATION ${BUILDPATH}/aie_api)

endfunction()
