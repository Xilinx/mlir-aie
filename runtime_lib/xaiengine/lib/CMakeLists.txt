# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Advanced Micro Devices, Inc.

include("../aiert.cmake")

set(XAIE_SOURCE ${VITIS_ROOT}/data/embeddedsw/XilinxProcessorIPLib/drivers/aienginev2_v3_0/src)

add_aiert_library(xaiengine ${XAIE_SOURCE})

if (${AIE_RUNTIME_TARGET} STREQUAL "aarch64")
target_compile_definitions(xaiengine PRIVATE __AIELINUX__)
endif()

install(TARGETS xaiengine DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/xaiengine/lib)
