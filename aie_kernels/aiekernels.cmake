function(add_aie_kernels_headers SRCPATH BUILDPATH INSTALLPATH)
  message("Installing aie_kernels includes from ${SRCPATH} in ${INSTALLPATH}")
  
  # copy header files into install area
  install(DIRECTORY ${SRCPATH}/ DESTINATION ${INSTALLPATH}/aie_kernels)
  
  # copy header files from install area to build area for testing
  add_custom_command(
    OUTPUT ${BUILDPATH}/aie_kernels
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${INSTALLPATH}/aie_kernels ${BUILDPATH}/aie_kernels
    DEPENDS ${INSTALLPATH}/aie_kernels
    COMMENT "Copying aie_kernels includes from ${INSTALLPATH} to ${BUILDPATH}"
  )

endfunction()
