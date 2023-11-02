#!/usr/bin/env bash
##===- utils/build-mlir-aie.sh - Build mlir-aie --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##
#
# This script builds mlir-aie given <llvm dir>.
# Assuming they are all in the same subfolder, it would look like:
#
# build-mlir-aie.sh <llvm dir> <build dir> <install dir>
#
# e.g. build-mlir-aie.sh /scratch/llvm
#
# <build dir>    - optional, mlir-aie/build dir name, default is 'build'
# <install dir>  - optional, mlir-aie/install dir name, default is 'install'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 1 ]; then
    echo "ERROR: Needs at least 1 arguments for <llvm build dir>."
    exit 1
fi

BASE_DIR=`realpath $(dirname $0)/..`
CMAKEMODULES_DIR=$BASE_DIR/cmake

LLVM_BUILD_DIR=`realpath $1`

BUILD_DIR=${3:-"build"}
INSTALL_DIR=${4:-"install"}

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR
set -o pipefail
set -e
cmake -GNinja\
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_DIR=${LLVM_BUILD_DIR}/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_BUILD_DIR}/lib/cmake/mlir \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/modulesXilinx \
    -DCMAKE_INSTALL_PREFIX="../${INSTALL_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    "-DAIE_RUNTIME_TARGETS=x86_64;aarch64" \
    -DAIE_RUNTIME_TEST_TARGET=aarch64 \
    .. 2>&1 | tee cmake.log

ninja 2>&1 | tee ninja.log
ninja install 2>&1 | tee ninja-install.log
#ninja check-aie 2>&1 | tee ninja-check-aie.log
