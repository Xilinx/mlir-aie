function(add_aie_kernels_headers SRCPATH BUILDPATH INSTALLPATH)
  message("Installing aie_kernels includes from ${SRCPATH} in ${INSTALLPATH}")
  
  # copy header files into install area
  install(DIRECTORY ${SRCPATH}/ DESTINATION ${INSTALLPATH}/aie_kernels)

  message("Copying aie_kernels includes from ${SRCPATH} to ${BUILDPATH}")
  
  # copy header files from install area to build area for testing
  file(COPY ${SRCPATH} DESTINATION ${BUILDPATH}/aie_kernels)

endfunction()
