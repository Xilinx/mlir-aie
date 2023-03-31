#!/usr/bin/env bash
##===- utils/build-llvm-local.sh - Build LLVM on local machine --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script build LLVM with custom options intended to be called on your
# machine where cloned llvm directory is in the current directory
#
# ./build-llvm-local.sh <llvm dir> <build dir> <install dir>
#
# <llvm dir>    - optional, default is 'llvm'
# <build dir>   - optional, default is 'build' (for llvm/build)
# <install dir> - optional, default is 'install' (for llvm/install)
#
##===----------------------------------------------------------------------===##

LLVM_DIR=${1:-"llvm"}
BUILD_DIR=${2:-"build"}
INSTALL_DIR=${3:-"install"}

mkdir -p $LLVM_DIR/$BUILD_DIR
mkdir -p $LLVM_DIR/$INSTALL_DIR
cd $LLVM_DIR/$BUILD_DIR
set -o pipefail
set -e
cmake ../llvm \
  -GNinja \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DPython3_FIND_VIRTUALENV=FIRST \
  -DLLVM_LINK_LLVM_DYLIB=ON \
  -DCLANG_LINK_CLANG_DYLIB=ON \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_BUILD_UTILS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_USE_LINKER=lld \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX=../$INSTALL_DIR \
  -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
  -DLLVM_TARGETS_TO_BUILD:STRING="X86;ARM;AArch64;" \
  -DCMAKE_BUILD_TYPE=Release \
  |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
cd ../..
