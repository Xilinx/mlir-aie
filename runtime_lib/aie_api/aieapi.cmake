
function(add_aie_api_headers SRCPATH BUILDPATH INSTALLPATH)
  message("Installing aie_api includes from ${SRCPATH} in ${BUILDPATH}")

  # copy header files into build area
  install(DIRECTORY ${SRCPATH}/ DESTINATION ${BUILDPATH}/aie_api)

  message("Installing aie_api includes from ${SRCPATH} in ${INSTALLPATH}")

  # install area too
  install(DIRECTORY ${SRCPATH}/ DESTINATION ${INSTALLPATH}/aie_api)

endfunction()
