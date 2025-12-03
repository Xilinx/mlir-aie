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
    VERSION=$(utils/clone-llvm.sh --get-wheel-version)

    mkdir -p my_install
    pushd my_install
    python3 -m pip -q download mlir==$VERSION \
      -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro
    unzip -q -u mlir-*.whl
    popd
    WHL_MLIR_DIR=`realpath my_install/mlir`
    echo "WHL_MLIR DIR: $WHL_MLIR_DIR"
else 
    WHL_MLIR_DIR=`realpath $1`
    echo "WHL_MLIR DIR: $WHL_MLIR_DIR"
fi

BASE_DIR=`realpath $(dirname $0)/..`
CMAKEMODULES_DIR=$BASE_DIR/cmake
echo "CMAKEMODULES_DIR: $CMAKEMODULES_DIR"

BUILD_DIR=${2:-"build"}
INSTALL_DIR=${3:-"install"}
LLVM_ENABLE_RTTI=${LLVM_ENABLE_RTTI:OFF}

if [ "$#" -ge 4 ]; then
  PEANO_INSTALL_DIR=`realpath $4`
  echo "PEANO_INSTALL_DIR DIR: $PEANO_INSTALL_DIR"
  export PEANO_INSTALL_DIR=${PEANO_INSTALL_DIR}
else
  python3 -m pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
  export PEANO_INSTALL_DIR="$(pip show llvm-aie | grep ^Location: | awk '{print $2}')/llvm-aie"
  echo "PEANO_INSTALL_DIR DIR: $PEANO_INSTALL_DIR"
  export PEANO_INSTALL_DIR=${PEANO_INSTALL_DIR}
fi

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
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DAIE_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_ENABLE_RTTI=$LLVM_ENABLE_RTTI \
    -DAIE_VITIS_COMPONENTS=AIE2;AIE2P \
    -DAIE_RUNTIME_TARGETS=x86_64 \
    -DAIE_RUNTIME_TEST_TARGET=x86_64 "

CMAKE_CONFIGS="${CMAKE_CONFIGS} -DPEANO_INSTALL_DIR=${PEANO_INSTALL_DIR}"

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
