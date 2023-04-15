#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

# cmake needs this line
cmake_minimum_required(VERSION 3.20.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project("aie runtime libs")

# include(ExternalProject)

# set(INSTALLS
#     test_library.cpp
#     test_library.h)

# foreach(file ${INSTALLS})
#     add_custom_target(aie-copy-runtime-libs-${file} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${file})
#     add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file}
#                     COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
#                     ${CMAKE_CURRENT_BINARY_DIR}/${file}
#                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
#     # add_dependencies(runtime-libs aie-copy-runtime-libs-${file} )
# endforeach()

# # Install too
# install(FILES ${INSTALLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib)

add_subdirectory(xaiengine)