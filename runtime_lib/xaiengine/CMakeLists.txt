# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD

# cmake needs this line
cmake_minimum_required(VERSION 3.20.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project("xaiengine driver")
set(XAIE_SOURCE ${VITIS_ROOT}/data/embeddedsw/XilinxProcessorIPLib/drivers/aienginev2_v3_2/src)
file(GLOB libsources ${XAIE_SOURCE}/*/*.c ${XAIE_SOURCE}/*/*/*.c)
file(GLOB libheaders ${VITIS_AIETOOLS_DIR}/include/drivers/aiengine/*.h)
file(GLOB libheadersSub ${VITIS_AIETOOLS_DIR}/include/drivers/aiengine/xaiengine/*.h)

include_directories(
    ${XAIE_SOURCE}
    ${XAIE_SOURCE}/common
    ${XAIE_SOURCE}/core
    ${XAIE_SOURCE}/device
    ${XAIE_SOURCE}/dma
    ${XAIE_SOURCE}/events
    ${XAIE_SOURCE}/global
    ${XAIE_SOURCE}/interrupt
    ${XAIE_SOURCE}/io_backend
    ${XAIE_SOURCE}/io_backend/ext
    ${XAIE_SOURCE}/io_backend/privilege
    ${XAIE_SOURCE}/lite
    ${XAIE_SOURCE}/locks
    ${XAIE_SOURCE}/memory
    ${XAIE_SOURCE}/npi
    ${XAIE_SOURCE}/perfcnt
    ${XAIE_SOURCE}/pl
    ${XAIE_SOURCE}/pm
    ${XAIE_SOURCE}/rsc
    ${XAIE_SOURCE}/stream_switch
    ${XAIE_SOURCE}/timer
    ${XAIE_SOURCE}/trace
    ${XAIE_SOURCE}/util
)

add_library(xaiengine SHARED ${libsources})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libxaiengine.so DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/xaiengine)
install(FILES ${libheaders} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/xaiengine/include)
install(FILES ${libheadersSub} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/xaiengine/include/xaiengine)