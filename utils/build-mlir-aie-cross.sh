#!/usr/bin/env bash
##===- utils/build-mlir-aie.sh - Build mlir-aie --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script build mlir-aie given the <sysroot dir>, <llvm dir>
# Assuming they are all in the same subfolder, it would look like:
#
# build-mlir-aie.sh <sysroot dir> <llvm dir> <build dir> <install dir>
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
cmake -GNinja \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/modulesXilinx \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKEMODULES_DIR}/toolchainFiles/toolchain_clang_crosscomp_pynq.cmake \
    -DSysroot=${SYSROOT_DIR} \
    -DArch=arm64 \
    -DLLVM_DIR=${LLVM_BUILD_DIR}/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_BUILD_DIR}/lib/cmake/mlir \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DAIE_ENABLE_BINDINGS_PYTHON=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -Wno-dev \
    .. |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
#ninja check-aie |& tee ninja-check-aie.log
cd ..
