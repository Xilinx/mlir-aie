#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

set(INSTALLS
    test_library.cpp
    test_library.h
    chess_intrinsic_wrapper.cpp)


# Stuff into the build area:
add_custom_target(copy-runtime-libs ALL)
# Extract me_basic.o from libme.a, which is shipped with Vitis.
add_custom_target(copy-runtime-libs-me_basic ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/me_basic.o)
    # Exists in the source tree.. just copy it.
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/me_basic.o
                    ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/me_basic.o)
elseif(DEFINED VITIS_ROOT)
    # Extract from Vitis.
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o
                       COMMAND ar -x ${VITIS_LIBME} me_basic.o)
else()
    message(WARNING "No source of me_basic.o... Hope you don't need it!")
endif()
add_dependencies(copy-runtime-libs copy-runtime-libs-me_basic )

# copy libc and libm from Vitis
if(DEFINED VITIS_ROOT)
    add_custom_target(copy-runtime-libs-libc ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libc.a)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libc.a
                    COMMAND ${CMAKE_COMMAND} -E copy ${VITIS_AIE_LIBC}
                    ${CMAKE_CURRENT_BINARY_DIR}/libc.a
                    DEPENDS ${VITIS_AIE_LIBC})
    add_dependencies(copy-runtime-libs copy-runtime-libs-libc )

    add_custom_target(copy-runtime-libs-libm ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libm.a)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libm.a
                    COMMAND ${CMAKE_COMMAND} -E copy ${VITIS_AIE_LIBM}
                    ${CMAKE_CURRENT_BINARY_DIR}/libm.a
                    DEPENDS ${VITIS_AIE_LIBM})
    add_dependencies(copy-runtime-libs copy-runtime-libs-libm )
    install(FILES ${VITIS_AIE_LIBC} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib)
    install(FILES ${VITIS_AIE_LIBM} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib)
else()
    message(WARNING "No source of libm/libc... Hope you don't need it!")
endif()

foreach(file ${INSTALLS})
    add_custom_target(copy-runtime-libs-${file} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${file})
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file}
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
                    ${CMAKE_CURRENT_BINARY_DIR}/${file}
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    add_dependencies(copy-runtime-libs copy-runtime-libs-${file} )
endforeach()

# Install too
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib)
install(FILES ${INSTALLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib)
