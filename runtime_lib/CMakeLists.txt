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
add_custom_target(aie-copy-runtime-libs ALL)
# Extract me_basic.o from libme.a, which is shipped with Vitis.
add_custom_target(me_basic ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o)
if(DEFINED VITIS_ROOT)
    # Compile me_basic.o
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o
                    COMMAND ${VITIS_XCHESSCC} -p me -P ${VITIS_AIETOOLS_DIR}/data/cervino/lib
                    -I ${VITIS_AIETOOLS_DIR}/data/cervino/lib/runtime/include/
                    -d -c ${CMAKE_CURRENT_SOURCE_DIR}/me_basic.c
                    -o ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/me_basic.c)
else()
    # Exists in the source tree.. just copy it.
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/me_basic.o
                    ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/me_basic.o)
endif()
add_dependencies(aie-copy-runtime-libs me_basic )

# copy libc and libm from Vitis
if(DEFINED VITIS_ROOT)
    add_custom_target(aie-copy-runtime-libs-libc ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libc.a)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libc.a
                    COMMAND ${CMAKE_COMMAND} -E copy ${VITIS_AIE_LIBC}
                    ${CMAKE_CURRENT_BINARY_DIR}/libc.a
                    DEPENDS ${VITIS_AIE_LIBC})
    add_dependencies(aie-copy-runtime-libs aie-copy-runtime-libs-libc )

    add_custom_target(aie-copy-runtime-libs-libm ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libm.a)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libm.a
                    COMMAND ${CMAKE_COMMAND} -E copy ${VITIS_AIE_LIBM}
                    ${CMAKE_CURRENT_BINARY_DIR}/libm.a
                    DEPENDS ${VITIS_AIE_LIBM})
    add_dependencies(aie-copy-runtime-libs aie-copy-runtime-libs-libm )
    install(FILES ${VITIS_AIE_LIBC} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib)
    install(FILES ${VITIS_AIE_LIBM} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib)
else()
    message(WARNING "No source of libm/libc... Hope you don't need it!")
endif()

foreach(file ${INSTALLS})
    add_custom_target(aie-copy-runtime-libs-${file} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${file})
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file}
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
                    ${CMAKE_CURRENT_BINARY_DIR}/${file}
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    add_dependencies(aie-copy-runtime-libs aie-copy-runtime-libs-${file} )
endforeach()

# Install too
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/me_basic.o DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib)
install(FILES ${INSTALLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib)
