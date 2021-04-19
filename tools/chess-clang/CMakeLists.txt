if(NOT DEFINED CMAKE_CROSSCOMPILE)
  set(SCRIPT_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/bin)
  # These files are only useful on host x86 compiles.
  set(FILES
    chess-clang
    xchesscc_wrapper)

  set(TARGETS ${FILES})
  list(TRANSFORM TARGETS PREPEND ${PROJECT_BINARY_DIR}/bin/)

  # stuff into the build directory
  add_custom_target(copy-scripts ALL DEPENDS ${TARGETS})
  foreach(file ${FILES})
  add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bin/${file}
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
                    ${PROJECT_BINARY_DIR}/bin/${file}
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
  endforeach()

  # install too!
  foreach(file ${FILES})
    install(PROGRAMS ${file} DESTINATION ${SCRIPT_INSTALL_PATH})
  endforeach()
endif()