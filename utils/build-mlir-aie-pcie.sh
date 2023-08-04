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
# build-mlir-aie.sh <llvm dir> <build dir> <install dir> <mlir-air-dir> <x86-libxaie-dir> <hsa-dir> <hsa-kmt-dir>
#
# e.g. build-mlir-aie.sh /scratch/llvm/build
#
# <build dir>    - optional, mlir-aie/build dir name, default is 'build'
# <install dir>  - optional, mlir-aie/install dir name, default is 'install'
# <x86-libxaie-dir> - optional, path to the x86 libxaie installation, necessary when compiling for VCK5000
# <HSA-dir> - optional, path to the HSA installation of the ROCm runtime
# <HSAKMT-dir> - optional, Path to the HSA KMT installation of the ROCm runtime
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 1 ]; then
    echo "ERROR: Needs at least 1 arguments for <llvm build dir>."
    exit 1
fi

BASE_DIR=`realpath $(dirname $0)/..`
CMAKEMODULES_DIR=$BASE_DIR/cmake

LLVM_BUILD_DIR=`realpath $1`
echo "LLVM BUILD DIR: $LLVM_BUILD_DIR"

BUILD_DIR=${2:-"build"}
INSTALL_DIR=${3:-"install"}
LIBXAIE_DIR=${4:-""}
HSA_DIR=${5:-""}
HSAKMT_DIR=${6:-""}
LLVM_ENABLE_RTTI=${LLVM_ENABLE_RTTI:OFF}


mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR
set -o pipefail
set -e

CMAKE_CONFIGS="\
    -GNinja \
    -DLLVM_DIR=${LLVM_BUILD_DIR}/lib/cmake/llvm \
    -DLibXAIE_x86_64-hsa_DIR=${LIBXAIE_DIR} \
    -Dhsa-runtime64_DIR=${HSA_DIR} \
    -Dhsakmt_DIR=${HSAKMT_DIR} \
    -DROCR_DIR=${ROCR_DIR} \
    -DMLIR_DIR=${LLVM_BUILD_DIR}/lib/cmake/mlir \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/modulesXilinx \
    -DCMAKE_INSTALL_PREFIX="../${INSTALL_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON \
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON \
    -DCMAKE_C_VISIBILITY_PRESET=hidden \
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=$LLVM_ENABLE_RTTI \
    -DAIE_RUNTIME_TARGETS=x86_64-hsa;aarch64 \
    -DAIE_ENABLE_PYTHON_PASSES=OFF \
    -DAIE_RUNTIME_TEST_TARGET=aarch64
    .. |& tee cmake.log"

if [ -x "$(command -v lld)" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DLLVM_USE_LINKER=lld"
fi

if [ -x "$(command -v ccache)" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DLLVM_CCACHE_BUILD=ON"
fi

cmake $CMAKE_CONFIGS .. 2>&1 | tee cmake.log
ninja 2>&1 | tee ninja.log
ninja install 2>&1 | tee ninja-install.log
