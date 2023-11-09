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

if [ "$#" -lt 1 ]; then
    echo "ERROR: Needs at least 1 arguments for <sysroot dir>"
    exit 1
fi

BASE_DIR=`realpath $(dirname $0)/..`
CMAKEMODULES_DIR=$BASE_DIR/cmake

SYSROOT_DIR=$1

LLVM_DIR=${2:-"llvm"}
BUILD_DIR=${3:-"${LLVM_DIR}/build-aarch64"}
INSTALL_DIR=${4:-"${LLVM_DIR}/install-aarch64"}

BUILD_DIR=`realpath ${BUILD_DIR}`
INSTALL_DIR=`realpath ${INSTALL_DIR}`
LLVM_DIR=`realpath ${LLVM_DIR}`

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR
set -o pipefail
set -e

CMAKE_CONFIGS="\
  -GNinja \
  -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/modulesXilinx \
  -DCMAKE_TOOLCHAIN_FILE=${CMAKEMODULES_DIR}/toolchainFiles/toolchain_clang_crosscomp_pynq.cmake \
  -DArch=arm64 \
  -DSysroot=${SYSROOT_DIR} \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_BUILD_UTILS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD:STRING=ARM;AArch64 \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_LINK_LLVM_DYLIB=OFF \
  -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON \
  -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON \
  -DCMAKE_C_VISIBILITY_PRESET=hidden \
  -DCMAKE_CXX_VISIBILITY_PRESET=hidden \
  -DMLIR_BINDINGS_PYTHON_ENABLED=ON \
  -DCMAKE_C_IMPLICIT_LINK_LIBRARIES=gcc_s \
  -DCMAKE_CXX_IMPLICIT_LINK_LIBRARIES=gcc_s \
  -DLLVM_ENABLE_PIC=True \
  -DMLIR_BUILD_UTILS=ON \
  -DMLIR_INCLUDE_TESTS=ON \
  -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF \
  -DLINKER_SUPPORTS_COLOR_DIAGNOSTICS=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_DEFAULT_TARGET_TRIPLE=aarch64-linux-gnu \
  -DLLVM_TARGET_ARCH=AArch64 \
  -Wno-dev"

if [ -x "$(command -v lld)" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DLLVM_USE_LINKER=lld"
fi

if [ -x "$(command -v ccache)" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DLLVM_CCACHE_BUILD=ON"
fi

cmake $CMAKE_CONFIGS ../llvm 2>&1 | tee cmake.log
ninja 2>&1 | tee ninja.log
ninja install 2>&1 | tee ninja-install.log
cd ../..
