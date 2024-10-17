#!/usr/bin/env bash
##===- utils/build-mlir-aie-from-wheels.sh - Build mlir-aie --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##
#
# This script builds mlir-aie given wheels installed <llvm mlir dir>.
# which is usually ./my_install/mlir if installed from the 
# ./utils/quick_setup.sh script.
# Assuming they are all in the same subfolder, it would look like:
#
# build-mlir-aie.sh <llvm mlir dir> <build dir> <install dir>
#
# e.g. build-mlir-aie-from-wheels.sh ./my_install/mlir 
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
echo "CMAKEMODULES_DIR: $CMAKEMODULES_DIR"

WHL_MLIR_DIR=`realpath $1`
echo "WHL_MLIR DIR: $WHL_MLIR_DIR"

BUILD_DIR=${2:-"build"}
INSTALL_DIR=${3:-"install"}
LLVM_ENABLE_RTTI=${LLVM_ENABLE_RTTI:OFF}

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR
#set -o pipefail
#set -e

CMAKE_CONFIGS="\
    -GNinja \
    -DCMAKE_PREFIX_PATH=${WHL_MLIR_DIR} \
    -DVITIS_VPP=$(which  v++) \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/modulesXilinx \
    -DLLVM_EXTERNAL_LIT=$(which lit) \
    -DCMAKE_INSTALL_PREFIX="../${INSTALL_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DAIE_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_ENABLE_RTTI=$LLVM_ENABLE_RTTI \
    -DAIE_RUNTIME_TARGETS=x86_64 \
    -DAIE_RUNTIME_TEST_TARGET=x86_64 "

if [ "$#" -eq 4 ]; then
  PEANO_INSTALL_DIR=`realpath $4`
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DPEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)"
fi 

if [ -x "$(command -v lld)" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DLLVM_USE_LINKER=lld"
fi

if [ -x "$(command -v ccache)" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DLLVM_CCACHE_BUILD=ON"
fi

cmake $CMAKE_CONFIGS .. 2>&1 | tee cmake.log
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
