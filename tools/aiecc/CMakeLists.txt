set(PYTHON_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/bin)

set(AIECC_SUBFILES
  cl_arguments.py
  __init__.py
  main.py)

set(AIECC_FILES
  aiecc.py
  aiecc/cl_arguments.py
  aiecc/__init__.py
  aiecc/main.py)

set(AIECC_TARGETS ${AIECC_FILES})
list(TRANSFORM AIECC_TARGETS PREPEND ${PROJECT_BINARY_DIR}/bin/)

# Stuff our python into the build area:
add_custom_target(aiecc.py ALL DEPENDS ${AIECC_TARGETS})
foreach(file ${AIECC_FILES})
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bin/${file}
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
                   ${PROJECT_BINARY_DIR}/bin/${file}
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

# Install our python stuff too.
install(PROGRAMS aiecc.py DESTINATION ${PYTHON_INSTALL_PATH})
foreach(file ${AIECC_SUBFILES})
  install(PROGRAMS aiecc/${file} DESTINATION ${PYTHON_INSTALL_PATH}/aiecc)
endforeach()
