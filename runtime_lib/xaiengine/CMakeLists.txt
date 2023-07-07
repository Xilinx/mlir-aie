# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Advanced Micro Devices, Inc.

cmake_minimum_required(VERSION 3.20.1)

#tmp path to header files since we use an older (3_0) version of xaiengine. need to fix when we upgrade to 2023.1
if(${Vitis_VERSION_MAJOR} EQUAL "2022")
    set(aieRTIncludePath "${VITIS_ROOT}/data/embeddedsw/XilinxProcessorIPLib/drivers/aienginev2_v3_0/src" CACHE STRING "AIE-RT include path")
elseif(${Vitis_VERSION_MAJOR} EQUAL "2023")
    set(aieRTIncludePath "${VITIS_AIETOOLS_DIR}/include/drivers/aiengine" CACHE STRING "AIE-RT include path")
else()
    message(FATAL_ERROR "Unsupported Vitis version: ${Vitis_VERSION_MAJOR}")
endif()

project("xaiengine lib for ${AIE_RUNTIME_TARGET}")

if (${CMAKE_CROSSCOMPILING})
    message("Building xaiengine for ${AIE_RUNTIME_TARGET} from Vitis at ${VITIS_ROOT}.") 
    file(GLOB libheaders ${aieRTIncludePath}/*.h)
    file(GLOB libheadersSub ${aieRTIncludePath}/*/*.h)

    # copy header files into build area
    foreach(file ${libheaders})
        cmake_path(GET file FILENAME basefile)
        # message("basefile: ${basefile}")
        set(dest ${CMAKE_CURRENT_BINARY_DIR}/include/${basefile})
        add_custom_target(aie-copy-runtime-libs-${basefile} ALL DEPENDS ${dest})
        add_custom_command(OUTPUT ${dest}
                        COMMAND ${CMAKE_COMMAND} -E copy ${file} ${dest}
                        DEPENDS ${file})
    endforeach()
    foreach(file ${libheadersSub})
        cmake_path(GET file FILENAME basefile)
        # message("basefile: ${basefile}")
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

else()
    message("Copying xaiengine for ${AIE_RUNTIME_TARGET} from Vitis at ${VITIS_ROOT}.") 
    set(xaiengineIncludePath ${VITIS_AIETOOLS_DIR}/include/drivers/aiengine)
    #tmp path to header files since we use an older (3_0) version of xaiengine. need to fix when we upgrade to 2023.1 and can take this from aie-rt
    set(xaiengineIncludePathTmp ${VITIS_ROOT}/data/embeddedsw/XilinxProcessorIPLib/drivers/aienginev2_v3_0/src)
    file(GLOB libheaders ${aieRTIncludePath}/*.h)
    file(GLOB libheadersSub ${aieRTIncludePath}/*/*.h)
    set(xaiengineLibPath ${VITIS_ROOT}/aietools/lib/lnx64.o/libxaiengine.so)

    # copy header files into build area
    set(dest ${CMAKE_CURRENT_BINARY_DIR}/include)
    add_custom_target(aie-copy-runtime-headers ALL DEPENDS ${dest})
    add_custom_command(OUTPUT ${dest}
                    COMMAND ${CMAKE_COMMAND} -E copy_directory ${xaiengineIncludePath} ${dest})

    # and the libraries
    set(dest ${CMAKE_CURRENT_BINARY_DIR}/lib/libxaiengine.so)
    add_custom_target(aie-copy-runtime-libs ALL DEPENDS ${dest})
    add_custom_command(OUTPUT ${dest}
                    COMMAND ${CMAKE_COMMAND} -E copy ${xaiengineLibPath} ${dest})

    install(FILES ${libheaders} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/xaiengine/include)
    install(FILES ${libheadersSub} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/xaiengine/include/xaiengine)
    install(FILES ${xaiengineLibPath}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIE_RUNTIME_TARGET}/xaiengine/lib)

    # /proj/xbuilds/2023.1_daily_latest/installs/lin64/Vitis/2023.1/aietools/include/drivers/aiengine
    #     ./aietools/lib/lnx64.o/libxaiengine.so
endif()
