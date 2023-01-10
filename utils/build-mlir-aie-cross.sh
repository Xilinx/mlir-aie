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
#     <mlir-aie dir> <build dir> <install dir>
#
# e.g. build-mlir-aie.sh /scratch/vck190_bare_prod_sysroot 10.2.0 /scratch/llvm 
#          /scratch/cmakeModules/cmakeModulesXilinx
#
# <sysroot dir>  - sysroot, absolute directory 
# <gcc version>  - gcc version in sysroot (needed in many petalinux sysroots to find imporant libs)
# <mlir-aie dir> - optional, mlir-aie repo name, default is 'mlir-aie'
# <build dir>    - optional, mlir-aie/build dir name, default is 'build'
# <install dir>  - optional, mlir-aie/install dir name, default is 'install'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 4 ]; then
    echo "ERROR: Needs at least 4 arguments for <sysroot dir>, <gcc version>, "
    echo "<llvm dir> and <cmakeModules dir>."
    exit 1
fi
SYSROOT_DIR=$1
GCC_VER=$2
LLVM_DIR=$3
CMAKEMODULES_DIR=$4

MLIR_AIE_DIR=${5:-"mlir-aie"}
BUILD_DIR=${6:-"build"}
INSTALL_DIR=${7:-"install"}

mkdir -p $MLIR_AIE_DIR/$BUILD_DIR
mkdir -p $MLIR_AIE_DIR/$INSTALL_DIR
cd $MLIR_AIE_DIR/$BUILD_DIR
set -o pipefail
set -e
cmake -GNinja \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKEMODULES_DIR}/toolchain_clang_crosscomp_arm_petalinux.cmake \
    -DSysroot=${SYSROOT_DIR} \
    -DArch=arm64 \
    -DgccVer=${GCC_VER} \
    -DLLVM_DIR=${LLVM_DIR}/build/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_DIR}/build/lib/cmake/mlir \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/ \
    -DCMAKE_INSTALL_PREFIX="../${INSTALL_DIR}" \
    -DVitisSysroot=${SYSROOT_DIR} \
    -DCMAKE_BUILD_TYPE=Debug \
    -DAIE_ENABLE_BINDINGS_PYTHON=ON \
    .. |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
#ninja check-aie |& tee ninja-check-aie.log
