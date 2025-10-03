#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices Inc.

function(install_headers SRCPATH BUILDPATH INSTALLPATH HEADERS_NAME)
  message("Installing ${HEADERS_NAME} includes from ${SRCPATH} in ${INSTALLPATH}/${HEADERS_NAME}")

  # copy header files into install area
  install(DIRECTORY ${SRCPATH}/ DESTINATION ${INSTALLPATH}/${HEADERS_NAME})

  message("Copying ${HEADERS_NAME} includes from ${SRCPATH} to ${BUILDPATH}/${HEADERS_NAME}")
  
  # copy header files into build area
  file(GLOB_RECURSE headers_to_copy ${SRCPATH}/*.h ${SRCPATH}/*.hpp)
  foreach(header ${headers_to_copy})
      file(RELATIVE_PATH rel_path ${SRCPATH} ${header})

      # create target name from file's path to avoid duplication
      get_filename_component(file_name "${header}" NAME)
      get_filename_component(parent_dir ${header} DIRECTORY)
      get_filename_component(module_name ${parent_dir} NAME)

      set(dest ${BUILDPATH}/${HEADERS_NAME}/${rel_path})
      add_custom_target(aie-copy-runtime-libs-${module_name}-${file_name} ALL DEPENDS ${dest})
      add_custom_command(OUTPUT ${dest}
                      COMMAND ${CMAKE_COMMAND} -E copy ${header} ${dest}
                      DEPENDS ${header}
      )
  endforeach()

endfunction()
