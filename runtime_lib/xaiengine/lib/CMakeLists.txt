# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Advanced Micro Devices, Inc.

include("../aiert.cmake")

if (${CMAKE_CROSSCOMPILING})
    message("Building xaiengine for ${AIE_RUNTIME_TARGET} from Vitis at ${VITIS_ROOT}.")

    #tmp path to header files since we use an older (3_0) version of xaiengine. need to fix when we upgrade to 2023.1
    if(${Vitis_VERSION_MAJOR} EQUAL "2022")
        set(XAIE_SOURCE ${VITIS_ROOT}/data/embeddedsw/XilinxProcessorIPLib/drivers/aienginev2_v3_0/src)
        set(aieRTIncludePath "${XAIE_SOURCE}")
    elseif(${Vitis_VERSION_MAJOR} EQUAL "2023")
        set(XAIE_SOURCE ${VITIS_ROOT}/data/embeddedsw/XilinxProcessorIPLib/drivers/aienginev2_v3_0/src)
        set(aieRTIncludePath "${VITIS_AIETOOLS_DIR}/include/drivers/aiengine")
    elseif(${Vitis_VERSION_MAJOR} EQUAL "2024")
        set(XAIE_SOURCE ${VITIS_ROOT}/data/embeddedsw/XilinxProcessorIPLib/drivers/aienginev2_v3_5/src)
        set(aieRTIncludePath "${VITIS_AIETOOLS_DIR}/include/drivers/aiengine")
    else()
        message(FATAL_ERROR "Unsupported Vitis version: ${Vitis_VERSION_MAJOR}")
    endif()

    message("Building xaiengine for ${AIE_RUNTIME_TARGET} from Vitis at ${VITIS_ROOT}.")
    add_aiert_headers(xaienginecdo
                        ${aieRTIncludePath}
                        ${CMAKE_CURRENT_BINARY_DIR}/include
                        ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/xaienginecdo/include)

    add_aiert_library(xaienginecdo ${XAIE_SOURCE})

    if (${AIE_RUNTIME_TARGET} STREQUAL "aarch64")
        target_compile_definitions(xaienginecdo PRIVATE __AIELINUX__)
    endif()
else()

    set(XAIE_SOURCE_DIR ../../../third_party/aie-rt/driver/src)
    set(BOOTGEN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/bootgen)

    add_aiert_headers(xaienginecdo
                        ${XAIE_SOURCE_DIR}
                        ${CMAKE_CURRENT_BINARY_DIR}/../include
                        ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/xaiengine/include)

    add_aiert_library(xaienginecdo ${XAIE_SOURCE_DIR})
    if (WIN32)
        target_compile_definitions(xaienginecdo PRIVATE __AIECDO__ )
    else()
        target_compile_definitions(xaienginecdo PRIVATE __AIECDO__ __AIESIM__)
    endif()
    target_include_directories(xaienginecdo SYSTEM PRIVATE ${BOOTGEN_SOURCE_DIR}/cdo-driver)
endif()

install(TARGETS xaienginecdo DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/xaiengine/lib)
