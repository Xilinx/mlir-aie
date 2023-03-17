#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Xilinx Inc.


set(Arch "arm64" CACHE STRING "ARM arch: arm64 or arm32")
set(pythonVer "3.8" CACHE STRING "python version in sysroot")
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES Sysroot Arch pythonVer)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR})

include(../modulesXilinx/toolchain_clang_crosscomp_arm)

# Pynq sysroot secifics 

set(GCC_INSTALL_PREFIX ${Sysroot}/usr CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "--gcc-toolchain=${CMAKE_SYSROOT}/usr" CACHE STRING "" FORCE)
