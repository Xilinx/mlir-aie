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

if [ "$#" -lt 2 ]; then
    echo "ERROR: Needs at least 2 arguments for <sysroot dir>, <cmakeModules dir>"
    exit 1
fi

SYSROOT_DIR=$1
CMAKEMODULES_DIR=`realpath $2`

LLVM_DIR=${3:-"llvm"}
BUILD_DIR=${5:-"${LLVM_DIR}/build-aarch64"}
INSTALL_DIR=${5:-"${LLVM_DIR}/install-aarch64"}

BUILD_DIR=`realpath ${BUILD_DIR}`
INSTALL_DIR=`realpath ${INSTALL_DIR}`
LLVM_DIR=`realpath ${LLVM_DIR}`

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR
set -o pipefail
set -e
cmake ../llvm \
  -GNinja \
  -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR} \
  -DCMAKE_TOOLCHAIN_FILE=${CMAKEMODULES_DIR}/toolchain_clang_crosscomp_arm.cmake \
  -DArch=arm64 \
  -DSysroot=${SYSROOT_DIR} \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_BUILD_UTILS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_USE_LINKER=lld \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
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
  -Wno-dev \
  |& tee cmake.log

#  -DCMAKE_BUILD_TYPE=MinSizeRel \
#  -DHAVE_POSIX_REGEX=0 \
#  -DHAVE_STEADY_CLOCK=0 \
#  -DLLVM_LINK_LLVM_DYLIB=ON \
#  -DCLANG_LINK_CLANG_DYLIB=ON \

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
cd ../..
