#
# Copyright (C) 2022 Xilinx, Inc.
# Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


set(INSTALLS
    genwrapper_for_ps.cpp)


foreach(file ${INSTALLS})
    add_custom_target(aie-copy-runtime-libs-AIE2-${file} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${file})
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file}
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
                    ${CMAKE_CURRENT_BINARY_DIR}/${file}
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    add_dependencies(aie-runtime-libs aie-copy-runtime-libs-AIE2-${file} )
endforeach()

install(FILES ${INSTALLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/aie_runtime_lib/AIE2/aiesim)

