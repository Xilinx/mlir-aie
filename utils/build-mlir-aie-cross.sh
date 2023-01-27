#!/usr/bin/env bash
##===- utils/build-mlir-aie.sh - Build mlir-aie --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script build mlir-aie given the <sysroot dir>, <llvm dir> and 
# <cmakeModules dir>. Assuming they are all in the same subfolder, it would
# look like:
#
# build-mlir-aie.sh <sysroot dir> <gcc version> <llvm dir> <cmakeModules dir> 
#     <build dir> <install dir>
#
# e.g. build-mlir-aie.sh /scratch/vck190_bare_prod_sysroot 10.2.0 /scratch/llvm 
#          /scratch/cmakeModules/cmakeModulesXilinx
#
# <sysroot dir>  - sysroot, absolute directory 
# <gcc version>  - gcc version in sysroot (needed in many petalinux sysroots to find imporant libs)
# <build dir>    - optional, mlir-aie/build dir name, default is 'build'
# <install dir>  - optional, mlir-aie/install dir name, default is 'install'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 3 ]; then
    echo "ERROR: Needs at least 4 arguments for <sysroot dir>, <llvm build dir> and <cmakeModules dir>."
    exit 1
fi

SYSROOT_DIR=$1

LLVM_BUILD_DIR=`realpath $2`
CMAKEMODULES_DIR=`realpath $3`

BUILD_DIR=${4:-"build-aarch64"}
INSTALL_DIR=${5:-"install-aarch64"}

BUILD_DIR=`realpath ${BUILD_DIR}`
INSTALL_DIR=`realpath ${INSTALL_DIR}`

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR
set -o pipefail
set -e
cmake -GNinja \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKEMODULES_DIR}/toolchain_clang_crosscomp_arm.cmake \
    -DSysroot=${SYSROOT_DIR} \
    -DArch=arm64 \
    -DLLVM_DIR=${LLVM_BUILD_DIR}/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_BUILD_DIR}/lib/cmake/mlir \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/ \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DVitisSysroot=${SYSROOT_DIR} \
    -DAIE_ENABLE_BINDINGS_PYTHON=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -Wno-dev \
    .. |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
#ninja check-aie |& tee ninja-check-aie.log
cd ..
