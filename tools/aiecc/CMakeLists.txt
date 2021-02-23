set(PYTHON_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/bin)

set(AIECC_FILES
  cl_arguments.py
  __init__.py
  main.py)

# Stuff our python into the build area:
configure_file(aiecc.py ${PROJECT_BINARY_DIR}/bin COPYONLY)
file(COPY aiecc DESTINATION ${PROJECT_BINARY_DIR}/bin)

# Install our python stuff too.
install(PROGRAMS aiecc.py DESTINATION ${PYTHON_INSTALL_PATH})
foreach(file ${AIECC_FILES})
  install(PROGRAMS aiecc/${file} DESTINATION ${PYTHON_INSTALL_PATH}/aiecc)
endforeach()

#configure_file("script2.py.in" "script2.py" @ONLY)

#set(SCRIPT2_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/<...>)
#install(FILES script2.py DESTINATION ${SCRIPT2_INSTALL_PATH}