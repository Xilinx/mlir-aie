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
# <target>      - optional, default is 'X86'
#
##===----------------------------------------------------------------------===##

LLVM_DIR=${1:-"llvm"}
BUILD_DIR=${2:-"build"}
INSTALL_DIR=${3:-"install"}
BUILD_TARGET=${4:-"X86"}

mkdir -p $LLVM_DIR/$BUILD_DIR
mkdir -p $LLVM_DIR/$INSTALL_DIR
LLVM_ENABLE_RTTI=${LLVM_ENABLE_RTTI:OFF}

# Enter a sub-shell to avoid messing up with current directory in case of error
(
cd $LLVM_DIR/$BUILD_DIR
set -o pipefail
set -e

CMAKE_CONFIGS="\
  -GNinja \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_LINK_LLVM_DYLIB=OFF \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_BUILD_UTILS=ON \
  -DLLVM_ENABLE_RTTI=$LLVM_ENABLE_RTTI \
  -DLLVM_INSTALL_UTILS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX=../$INSTALL_DIR \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD:STRING=$BUILD_TARGET \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON \
  -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON \
  -DCMAKE_C_VISIBILITY_PRESET=hidden \
  -DCMAKE_CXX_VISIBILITY_PRESET=hidden \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1"

if [ -x "$(command -v lld)" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DLLVM_USE_LINKER=lld"
fi

if [ -x "$(command -v ccache)" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DLLVM_CCACHE_BUILD=ON"
fi

cmake $CMAKE_CONFIGS ../llvm 2>&1 | tee cmake.log
ninja 2>&1 | tee ninja.log
ninja install 2>&1 | tee ninja-install.log
)
