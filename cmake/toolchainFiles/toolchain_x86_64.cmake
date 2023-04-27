# Copyright (C) 2018-2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# specify the compiler
set(CLANG_VER 12)
set(CMAKE_C_COMPILER clang-${CLANG_VER})
set(CMAKE_CXX_COMPILER clang++-${CLANG_VER})
set(CMAKE_ASM_COMPILER clang-${CLANG_VER})
set(CMAKE_STRIP llvm-strip-${CLANG_VER})
set(CLANG_LLD lld-${CLANG_VER} CACHE STRING "" FORCE)

# Make it a debug runtime build
#set(CMAKE_BUILD_TYPE Debug CACHE STRING "build type" FORCE)
#set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "" FORCE)

# the default for x86 is to build a PCIe runtime
set(BUILD_AIR_PCIE ON CACHE BOOL "" FORCE)
