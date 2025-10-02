function(add_aie_kernels_headers SRCPATH BUILDPATH INSTALLPATH)
  message("Installing aie_kernels includes from ${SRCPATH} in ${INSTALLPATH}")
  
  # copy aie_kernels header files into install area
  install(DIRECTORY ${SRCPATH}/ DESTINATION ${INSTALLPATH}/aie_kernels)

  message("Copying aie_kernels includes from ${SRCPATH} to ${BUILDPATH}/aie_kernels")
  
  # copy aie_kernels header files into build area
  file(GLOB_RECURSE headers_to_copy "${SRCPATH}/*.h" "${SRCPATH}/*.hpp")
  foreach(header ${headers_to_copy})
      file(RELATIVE_PATH rel_path ${SRCPATH} ${header})

      # create target name from file's path to avoid duplication
      get_filename_component(file_name "${header}" NAME)
      get_filename_component(parent_dir ${header} DIRECTORY)
      get_filename_component(module_name ${parent_dir} NAME)

      set(dest ${BUILDPATH}/aie_kernels/${rel_path})
      add_custom_target(aie-copy-runtime-libs-${module_name}-${file_name} ALL DEPENDS ${dest})
      add_custom_command(OUTPUT ${dest}
                      COMMAND ${CMAKE_COMMAND} -E copy ${header} ${dest}
                      DEPENDS ${header}
      )
  endforeach()

endfunction()
