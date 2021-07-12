#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

set(INSTALLS
    me_basic.o
    test_library.cpp
    test_library.h
    chess_intrinsic_wrapper.cpp)

# Stuff into the build area:
add_custom_target(copy-runtime-libs ALL)
foreach(file ${INSTALLS})
    add_custom_target(copy-runtime-libs-${file} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${file})
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file}
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
                    ${CMAKE_CURRENT_BINARY_DIR}/${file}
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    add_dependencies(copy-runtime-libs copy-runtime-libs-${file} )
endforeach()

# Install too
install(FILES ${INSTALLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib)
