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
# build-mlir-aie.sh <llvm build dir> <build dir> <install dir>
#
# e.g. build-mlir-aie.sh /scratch/llvm/build
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

BUILD_DIR=${2:-"build"}
INSTALL_DIR=${3:-"install"}

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR

##
## build pynqMLIR-AIE
##

# configure
mkdir -p ${BUILD_DIR}/mlir-aie
pushd ${BUILD_DIR}/mlir-aie 

CMAKE_CONFIGS="\
    -GNinja \
    -DLLVM_DIR=${LLVM_BUILD_DIR}/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_BUILD_DIR}/lib/cmake/mlir \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/modulesXilinx \
    -DCMAKE_INSTALL_PREFIX="../../${INSTALL_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DAIE_ENABLE_BINDINGS_PYTHON=ON \
    -DAIE_RUNTIME_TARGETS:STRING="x86_64" \
    -DAIE_RUNTIME_TEST_TARGET=x86_64 \
    ../.."

if [ -x "$(command -v lld)" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DLLVM_USE_LINKER=lld"
fi

if [ -x "$(command -v ccache)" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DLLVM_CCACHE_BUILD=ON"
fi

cmake $CMAKE_CONFIGS  2>&1 | tee cmake.log
ninja 2>&1 | tee mlir-aie-ninja.log
ninja install 2>&1 | tee mlir-aie-ninja-install.log
success=$?

if [ ${success} -ne 0 ]
then
    rm -rf ${INSTALL_DIR}/*
    rm -rf ${BUILD_DIR}/*
    popd; exit 4
fi

exit 0
