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
file(COPY ${VITIS_ROOT}/data/embeddedsw/XilinxProcessorIPLib/drivers/aienginev2_v3_2/src DESTINATION ${PROJECT_SOURCE_DIR})
file(GLOB libsources ${PROJECT_SOURCE_DIR}/src/*/*.c ${PROJECT_SOURCE_DIR}/src/*/*/*.c)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/core
    ${PROJECT_SOURCE_DIR}/src/device
    ${PROJECT_SOURCE_DIR}/src/dma
    ${PROJECT_SOURCE_DIR}/src/events
    ${PROJECT_SOURCE_DIR}/src/global
    ${PROJECT_SOURCE_DIR}/src/interrupt
    ${PROJECT_SOURCE_DIR}/src/io_backend
    ${PROJECT_SOURCE_DIR}/src/io_backend/ext
    ${PROJECT_SOURCE_DIR}/src/io_backend/privilege
    ${PROJECT_SOURCE_DIR}/src/lite
    ${PROJECT_SOURCE_DIR}/src/locks
    ${PROJECT_SOURCE_DIR}/src/memory
    ${PROJECT_SOURCE_DIR}/src/npi
    ${PROJECT_SOURCE_DIR}/src/perfcnt
    ${PROJECT_SOURCE_DIR}/src/pl
    ${PROJECT_SOURCE_DIR}/src/pm
    ${PROJECT_SOURCE_DIR}/src/rsc
    ${PROJECT_SOURCE_DIR}/src/stream_switch
    ${PROJECT_SOURCE_DIR}/src/timer
    ${PROJECT_SOURCE_DIR}/src/trace
    ${PROJECT_SOURCE_DIR}/src/util
)

add_library(xaiengine SHARED ${libsources})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libxaiengine.so DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/libxaie)