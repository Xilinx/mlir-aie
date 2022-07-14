#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

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

if(${AIE_COMPILER} STREQUAL XCHESSCC)
set(CONFIG_COMPILE_WITH_XCHESSCC True)
else()
set(CONFIG_COMPILE_WITH_XCHESSCC False)
endif()

if(${AIE_COMPILER} STREQUAL NONE)
set(CONFIG_DISABLE_COMPILE True)
else()
set(CONFIG_DISABLE_COMPILE False)
endif()

if(${AIE_LINKER} STREQUAL XCHESSCC)
set(CONFIG_LINK_WITH_XCHESSCC True)
else()
set(CONFIG_LINK_WITH_XCHESSCC False)
endif()

if(${AIE_LINKER} STREQUAL NONE)
set(CONFIG_DISABLE_LINK True)
else()
set(CONFIG_DISABLE_LINK False)
endif()

configure_file(aiecc/configure.py.in ${PROJECT_BINARY_DIR}/bin/aiecc/configure.py)
install(PROGRAMS ${PROJECT_BINARY_DIR}/bin/aiecc/configure.py DESTINATION ${PYTHON_INSTALL_PATH}/aiecc)

# Stuff our python into the build area:
add_custom_target(aiecc.py ALL DEPENDS ${AIECC_TARGETS})

# This chicanery is necessary to ensure executable permissions.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/copy_aiecc.cmake"
"file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/aiecc.py
DESTINATION ${PROJECT_BINARY_DIR}/bin
FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_WRITE
GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)")

add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bin/aiecc.py COMMAND
${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/copy_aiecc.cmake)

foreach(file ${AIECC_SUBFILES})
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bin/aiecc/${file}
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/aiecc/${file}
                   ${PROJECT_BINARY_DIR}/bin/aiecc/${file}
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/aiecc/${file})
endforeach()

# Install our python stuff too.
install(PROGRAMS aiecc.py DESTINATION ${PYTHON_INSTALL_PATH})
foreach(file ${AIECC_SUBFILES})
  install(PROGRAMS aiecc/${file} DESTINATION ${PYTHON_INSTALL_PATH}/aiecc)
endforeach()
