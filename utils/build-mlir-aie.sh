#!/usr/bin/env bash
##===- utils/build-mlir-aie.sh - Build mlir-aie --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script builds mlir-aie given the <llvm dir> and <cmakeModules dir>.
# Assuming they are all in the same subfolder, it would look like:
#
# build-mlir-aie.sh <llvm dir> <cmakeModules dir> <build dir> <install dir>
#
# e.g. build-mlir-aie.sh /scratch/llvm /scratch/cmakeModules/cmakeModulesXilinx
#
# <build dir>    - optional, mlir-aie/build dir name, default is 'build'
# <install dir>  - optional, mlir-aie/install dir name, default is 'install'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 2 ]; then
    echo "ERROR: Needs at least 2 arguments for <llvm dir> and <cmakeModules dir>."
    exit 1
fi
LLVM_DIR=$1
CMAKEMODULES_DIR=$2

BUILD_DIR=${3:-"build"}
INSTALL_DIR=${4:-"install"}

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR
set -o pipefail
set -e
cmake -GNinja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_DIR=${LLVM_DIR}/build/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_DIR}/build/lib/cmake/mlir \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/ \
    -DCMAKE_INSTALL_PREFIX="../${INSTALL_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DAIE_ENABLE_BINDINGS_PYTHON=ON \
    .. |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
#ninja check-aie |& tee ninja-check-aie.log
cd ..
