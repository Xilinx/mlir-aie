# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Advanced Micro Devices, Inc.

cmake_minimum_required(VERSION 3.20.1)

project("xaiengine lib for ${AIE_RUNTIME_TARGET}")

set(xaienginePath ${VITIS_AIETOOLS_DIR}/include/drivers/aiengine)
file(GLOB libheaders ${xaienginePath}/*.h)
file(GLOB libheadersSub ${xaienginePath}/xaiengine/*.h)

# copy header files into build area
foreach(file ${libheaders})
    file(RELATIVE_PATH basefile ${xaienginePath} ${file})
    message("basefile: ${basefile}")
    set(dest ${CMAKE_CURRENT_BINARY_DIR}/include/${basefile})
    add_custom_target(aie-copy-runtime-libs-${basefile} ALL DEPENDS ${dest})
    add_custom_command(OUTPUT ${dest}
                    COMMAND ${CMAKE_COMMAND} -E copy ${file} ${dest}
                    DEPENDS ${file})
endforeach()
foreach(file ${libheadersSub})
    file(RELATIVE_PATH basefile ${xaienginePath}/xaiengine ${file})
    message("basefile: ${basefile}")
    set(dest ${CMAKE_CURRENT_BINARY_DIR}/include/xaiengine/${basefile})
    add_custom_target(aie-copy-runtime-libs-${basefile} ALL DEPENDS ${dest})
    add_custom_command(OUTPUT ${dest}
                    COMMAND ${CMAKE_COMMAND} -E copy ${file} ${dest}
                    DEPENDS ${file})
endforeach()

# Install too
install(FILES ${libheaders} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/xaiengine/include)
install(FILES ${libheadersSub} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/xaiengine/include/xaiengine)

add_subdirectory(lib)
