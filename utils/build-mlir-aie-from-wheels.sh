#!/usr/bin/env bash
##===- utils/build-mlir-aie-from-wheels.sh - Build mlir-aie --*- Script -*-===##
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##===----------------------------------------------------------------------===##
#
# This script builds mlir-aie given wheels installed <llvm mlir dir>.
# which is usually ./my_install/mlir if installed from the
# ./utils/env_install.sh script.
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

# Windows (Git-for-Windows/MSYS) bash: gate off Linux-only assumptions.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=1 ;;
  *) IS_WINDOWS=0 ;;
esac

# On Windows rewrite MSYS /c/... to CMake's C:/... form; pass through on Linux.
winpath() {
  if [ "$IS_WINDOWS" = "1" ] && [ -n "$1" ]; then cygpath -m "$1"; else printf '%s' "$1"; fi
}

# Stamp unpacked wheel files to a fixed past time (see below).
stamp_mlir_past() {
  if [ "$IS_WINDOWS" = "1" ]; then
    # find -exec touch blows MSYS's env-block limit; one python pass instead.
    python -c "import os, pathlib; ts = 1314108314; [os.utime(p, (ts, ts)) for p in pathlib.Path('mlir').rglob('*')]"
  else
    find mlir -exec touch -a -m -t 201108231405.14 {} \;
  fi
}

if [ "$#" -lt 1 ] || [ -z "$1" ]; then
    VERSION=$(utils/clone-llvm.sh --get-wheel-version)

    mkdir -p my_install
    pushd my_install
    python3 -m pip -q download mlir==$VERSION \
      -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro
    unzip -q -u mlir-*.whl
    # The system clock on GHA containers is sometimes ~12 hours ahead, so the
    # freshly unzipped wheel ends up with files whose timestamps are in the
    # future. That makes ninja loop forever when configuring. Stamp them to an
    # arbitrary time in the past to be safe.
    stamp_mlir_past
    popd
    WHL_MLIR_DIR=$(winpath "$(realpath my_install/mlir)")
    echo "WHL_MLIR DIR: $WHL_MLIR_DIR"
else
    WHL_MLIR_DIR=$(winpath "$(realpath "$1")")
    echo "WHL_MLIR DIR: $WHL_MLIR_DIR"
fi

BASE_DIR=`realpath $(dirname $0)/..`
CMAKEMODULES_DIR=$(winpath "$BASE_DIR/cmake")
echo "CMAKEMODULES_DIR: $CMAKEMODULES_DIR"

BUILD_DIR=${2:-"build"}
INSTALL_DIR=${3:-"install"}
LLVM_ENABLE_RTTI=${LLVM_ENABLE_RTTI:-OFF}

if [ "$#" -ge 4 ]; then
  PEANO_INSTALL_DIR=$(winpath "$(realpath "$4")")
  echo "PEANO_INSTALL_DIR DIR: $PEANO_INSTALL_DIR"
  export PEANO_INSTALL_DIR=${PEANO_INSTALL_DIR}
else
  PEANO_LOCATION=$(pip show llvm-aie 2>/dev/null | awk '/^Location:/ {print $2}')
  if [ -z "$PEANO_LOCATION" ]; then
    echo "ERROR: llvm-aie (Peano) is not installed; the Python environment isn't set up." >&2
    echo "       Install it first with: source utils/env_install.sh <venv-dir> [--dev]" >&2
    echo "       Or pass an existing Peano install dir as the 4th argument to this script." >&2
    exit 1
  fi
  export PEANO_INSTALL_DIR=$(winpath "${PEANO_LOCATION}/llvm-aie")
  echo "PEANO_INSTALL_DIR DIR: $PEANO_INSTALL_DIR"
fi

BUILD_TYPE="${BUILD_TYPE:-Release}"

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR
#set -o pipefail
#set -e

# Build the -D list as an array so paths with spaces survive as single args.
CMAKE_CONFIGS=(
    -GNinja
    -DCMAKE_PREFIX_PATH="${WHL_MLIR_DIR}"
    -DCMAKE_MODULE_PATH="${CMAKEMODULES_DIR}/modulesXilinx"
    -DLLVM_EXTERNAL_LIT="$(winpath "$(which lit)")"
    -DCMAKE_INSTALL_PREFIX="../${INSTALL_DIR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DLLVM_ENABLE_ASSERTIONS=ON
    -DAIE_ENABLE_BINDINGS_PYTHON=ON
    -DLLVM_ENABLE_RTTI="$LLVM_ENABLE_RTTI"
    "-DAIE_VITIS_COMPONENTS=AIE2;AIE2P"
    -DAIE_RUNTIME_TARGETS=x86_64
    -DAIE_RUNTIME_TEST_TARGET=x86_64
    -DPEANO_INSTALL_DIR="${PEANO_INSTALL_DIR}"
)

# Vitis (v++) is Linux-only; skip the flag when it's absent (e.g. Windows).
if command -v v++ >/dev/null 2>&1; then
  CMAKE_CONFIGS+=(-DVITIS_VPP="$(winpath "$(which v++)")")
fi

if [ -x "$(command -v lld)" ]; then
  CMAKE_CONFIGS+=(-DLLVM_USE_LINKER=lld)
fi

if [ -x "$(command -v ccache)" ]; then
  CMAKE_CONFIGS+=(-DLLVM_CCACHE_BUILD=ON)
fi

# Allow callers (e.g. CI) to inject additional -D options without editing this
# script. Appended last so they can override the defaults above. Values with
# spaces can't survive this flat env var; callers must pass space-free paths.
if [ -n "${EXTRA_CMAKE_ARGS}" ]; then
  read -ra _extra_cmake_args <<< "${EXTRA_CMAKE_ARGS}"
  CMAKE_CONFIGS+=("${_extra_cmake_args[@]}")
fi

cmake "${CMAKE_CONFIGS[@]}" .. 2>&1 | tee cmake.log
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
