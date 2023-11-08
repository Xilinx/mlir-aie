#!/usr/bin/env bash
##===- utils/build-mlir-aie.sh - Build mlir-aie --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script build mlir-aie given the <sysroot dir>, <llvm build dir>
# Assuming they are all in the same subfolder, it would look like:
#
# build-mlir-aie.sh <sysroot dir> <llvm build dir> <build dir> <install dir>
#
# e.g. build-mlir-aie.sh /scratch/vck190_bare_prod_sysroot 10.2.0 /scratch/llvm 
#          /scratch/cmakeModules/cmakeModulesXilinx
#
# <sysroot dir>  - sysroot, absolute directory 
# <build dir>    - optional, mlir-aie/build dir name, default is 'build'
# <install dir>  - optional, mlir-aie/install dir name, default is 'install'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 2 ]; then
    echo "ERROR: Needs at least 2 arguments for <sysroot dir>, <llvm build dir>"
    exit 1
fi

BASE_DIR=`realpath $(dirname $0)/..`
CMAKEMODULES_DIR=$BASE_DIR/cmake

SYSROOT_DIR=$1

LLVM_BUILD_DIR=`realpath $2`

BUILD_DIR=${3:-"build-aarch64"}
INSTALL_DIR=${4:-"install-aarch64"}

BUILD_DIR=`realpath ${BUILD_DIR}`
INSTALL_DIR=`realpath ${INSTALL_DIR}`

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR
set -o pipefail
set -e

CMAKE_CONFIGS="\
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/modulesXilinx \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKEMODULES_DIR}/toolchainFiles/toolchain_clang_crosscomp_pynq.cmake \
    -DSysroot=${SYSROOT_DIR} \
    -DArch=arm64 \
    -DLLVM_DIR=${LLVM_BUILD_DIR}/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_BUILD_DIR}/lib/cmake/mlir \
    -DLLVM_ENABLE_RTTI=ON \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON \
    -DCMAKE_C_VISIBILITY_PRESET=hidden \
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden \
    -Wno-dev"

if [ -x "$(command -v lld)" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DLLVM_USE_LINKER=lld"
fi

if [ -x "$(command -v ccache)" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DLLVM_CCACHE_BUILD=ON"
fi

cmake $CMAKE_CONFIGS 2>&1 | tee cmake.log
ninja 2>&1 | tee ninja.log
ninja install 2>&1 | tee ninja-install.log
cd ..
