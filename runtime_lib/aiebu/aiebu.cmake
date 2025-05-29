# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

function(add_aiebu SRCPATH BUILDPATH INSTALLPATH)

  # Add a custom target to build AIEBU
  add_custom_command(
    OUTPUT ${SRCPATH}/build/Release/AIEBU-1.0-Linux.tar.gz
    COMMAND ${CMAKE_COMMAND} -E echo "Building AIEBU in ${SRCPATH} ${BUILDPATH}"
    COMMAND ${CMAKE_COMMAND} -E chdir ${SRCPATH}/build ./build.sh
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BUILDPATH}
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${BUILDPATH}/aiebu
    COMMAND tar -xzf ${SRCPATH}/build/Release/AIEBU-1.0-Linux.tar.gz -C ${BUILDPATH}
    COMMAND ${CMAKE_COMMAND} -E rename ${BUILDPATH}/AIEBU-1.0-Linux ${BUILDPATH}/aiebu
    WORKING_DIRECTORY ${SRCPATH}
    DEPENDS ${SRCPATH}
    COMMENT "Building AIEBU"
  )

  # Add a custom target to install AIEBU
  add_custom_target(install_aiebu ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Installing AIEBU to ${INSTALLPATH}"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTALLPATH}
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${INSTALLPATH}/aiebu
    COMMAND tar -xzf ${SRCPATH}/build/Release/AIEBU-1.0-Linux.tar.gz -C ${INSTALLPATH}
    COMMAND ${CMAKE_COMMAND} -E rename ${INSTALLPATH}/AIEBU-1.0-Linux ${INSTALLPATH}/aiebu
    DEPENDS ${SRCPATH}/build/Release/AIEBU-1.0-Linux.tar.gz
    COMMENT "Installing AIEBU"
  )

endfunction()
