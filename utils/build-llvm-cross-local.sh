#!/usr/bin/env bash
##===- utils/build-llvm-local.sh - Build LLVM on local machine --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script build LLVM with custom options intended to be called on your
# machine where cloned llvm directory is in the current directory
#
# ./build-llvm-local.sh <sysroot dir> <gcc version> <cmakeModules dir> <llvm dir> 
# <build dir> <install dir>
#
# <sysroot dir> - sysroot, absolute directory 
# <gcc version> - gcc version in sysroot (needed in many petalinux sysroots to find 
#                 imporant libs)
# <cmakeModules dir> - cmakeModules, absolute directory 
#                      (usually cmakeModules/cmakeModulesXilinx)
# <llvm dir>    - optional, default is 'llvm'
# <build dir>   - optional, default is 'build' (for llvm/build)
# <install dir> - optional, default is 'install' (for llvm/install)
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 3 ]; then
    echo "ERROR: Needs at least 2 arguments for <sysroot dir>, <gcc version>,"
    echo "<cmakeModules dir>."
    exit 1
fi

SYSROOT_DIR=$1
GCC_VER=$2
CMAKEMODULES_DIR=$3
LLVM_DIR=${4:-"llvm"}
BUILD_DIR=${5:-"build"}
INSTALL_DIR=${6:-"install"}

mkdir -p $LLVM_DIR/$BUILD_DIR
mkdir -p $LLVM_DIR/$INSTALL_DIR
cd $LLVM_DIR/$BUILD_DIR
set -o pipefail
set -e
cmake ../llvm \
  -GNinja \
  -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR} \
  -DCMAKE_TOOLCHAIN_FILE=${CMAKEMODULES_DIR}/toolchain_clang_crosscomp_arm_petalinux.cmake \
  -DArch=arm64 \
  -DgccVer=${GCC_VER} \
  -DSysroot=${SYSROOT_DIR} \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_BUILD_UTILS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_USE_LINKER=lld \
  -DCMAKE_INSTALL_PREFIX=../$INSTALL_DIR \
  -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
  -DLLVM_TARGETS_TO_BUILD:STRING="ARM;AArch64" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_LINK_LLVM_DYLIB=OFF \
  -DCLANG_LINK_CLANG_DYLIB=OFF \
  -DMLIR_BINDINGS_PYTHON_ENABLED=ON \
  -DCMAKE_C_IMPLICIT_LINK_LIBRARIES=gcc_s \
  -DCMAKE_CXX_IMPLICIT_LINK_LIBRARIES=gcc_s \
  -DLLVM_ENABLE_PIC=True \
  -DMLIR_BUILD_UTILS=ON \
  -DMLIR_INCLUDE_TESTS=ON \
  -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF \
  -DLINKER_SUPPORTS_COLOR_DIAGNOSTICS=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_DEFAULT_TARGET_TRIPLE="aarch64-linux-gnu" \
  -DLLVM_TARGET_ARCH="AArch64" \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  |& tee cmake.log

#  -DCMAKE_BUILD_TYPE=MinSizeRel \
#  -DHAVE_POSIX_REGEX=0 \
#  -DHAVE_STEADY_CLOCK=0 \
#  -DLLVM_LINK_LLVM_DYLIB=ON \
#  -DCLANG_LINK_CLANG_DYLIB=ON \

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
cd ../..
