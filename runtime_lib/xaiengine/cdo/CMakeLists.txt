# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Advanced Micro Devices, Inc.

include("../aiert.cmake")

message("Building xaiengine for ${AIE_RUNTIME_TARGET} for CDO backend.")

set(XAIE_SOURCE ../../../third_party/aie-rt/driver/src)

add_aiert_headers(xaienginecdo
                    ${XAIE_SOURCE}
                    ${CMAKE_CURRENT_BINARY_DIR}/include
                    ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/xaiengine/cdo/include)

add_aiert_library(xaienginecdo ${XAIE_SOURCE})

target_compile_definitions(xaienginecdo PRIVATE -D__AIECDO__)

install(TARGETS xaienginecdo DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/xaiengine/cdo)
