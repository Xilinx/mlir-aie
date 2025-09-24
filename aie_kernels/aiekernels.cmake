function(add_aie_kernels_headers SRCPATH BUILDPATH INSTALLPATH)
  message("Installing aie_kernels includes from ${SRCPATH} in ${BUILDPATH}")

  # copy header files into build area
  install(DIRECTORY ${SRCPATH}/ DESTINATION ${BUILDPATH}/aie_kernels)

  message("Installing aie_kernels includes from ${SRCPATH} in ${INSTALLPATH}")

  # install area too
  install(DIRECTORY ${SRCPATH}/ DESTINATION ${INSTALLPATH}/aie_kernels)

endfunction()
