#!/bin/bash
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
##===- utils/env_setup.sh - Setup mlir-aie env to compile IRON designs --*- Script -*-===##
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Configure the current shell to build and run IRON designs against an already
# installed mlir_aie + Peano (see utils/env_install.sh). This script only sets
# environment variables and inspects the machine; it never installs anything.
#
#   source env_setup.sh [<mlir-aie install dir> [<llvm-aie/peano install dir>]]
#
# With no arguments the install locations are discovered from the active Python
# environment (pip show mlir_aie / llvm-aie). If a dependency is missing the
# script explains how to install it and returns non-zero.
#
# e.g. source env_setup.sh install /scratch/llvm-aie/install
#
##===----------------------------------------------------------------------===##

# --- MLIR-AIE install dir ----------------------------------------------------
# Resolve the repo root from the script's own location so a from-source ("dev")
# build is detected regardless of the caller's working directory.
_env_setup_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$#" -ge 1 ]; then
    export MLIR_AIE_INSTALL_DIR=`realpath $1`
elif [ -d "$_env_setup_repo_root/install" ]; then
    # Dev build: a from-source build installs into an in-tree install/ dir.
    export MLIR_AIE_INSTALL_DIR="$_env_setup_repo_root/install"
    echo "Using in-tree dev build: $MLIR_AIE_INSTALL_DIR"
else
    MLIR_AIE_INSTALL_DIR="$(pip show mlir_aie 2>/dev/null | awk '/^Location:/ {print $2 "/mlir_aie"}')"
    if [ -z "$MLIR_AIE_INSTALL_DIR" ]; then
        echo "ERROR: mlir_aie is not installed in the active Python environment." >&2
        echo "       Install it first, e.g.: source utils/env_install.sh" >&2
        echo "       or pass its install dir:  source utils/env_setup.sh <mlir-aie-dir> [peano-dir]" >&2
        unset _env_setup_repo_root
        return 1
    fi
    export MLIR_AIE_INSTALL_DIR
fi
unset _env_setup_repo_root

# --- Peano (llvm-aie) install dir --------------------------------------------
if [ "$#" -ge 2 ]; then
    export PEANO_INSTALL_DIR=`realpath $2`
else
    PEANO_INSTALL_DIR="$(pip show llvm-aie 2>/dev/null | awk '/^Location:/ {print $2 "/llvm-aie"}')"
    if [ -z "$PEANO_INSTALL_DIR" ]; then
        echo "ERROR: llvm-aie (Peano) is not installed in the active Python environment." >&2
        echo "       Install it first, e.g.: source utils/env_install.sh" >&2
        echo "       or pass its install dir:  source utils/env_setup.sh <mlir-aie-dir> <peano-dir>" >&2
        return 1
    fi
    export PEANO_INSTALL_DIR
fi

export PATH=${MLIR_AIE_INSTALL_DIR}/bin:${PATH}
export PYTHONPATH=${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH}
export LD_LIBRARY_PATH=${MLIR_AIE_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}

# --- NPU detection (native + WSL) --------------------------------------------
NPUPAT='NPU Phoenix|NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[1456]'
if [ -n "${WSL_DISTRO_NAME-}" ]; then
    # Under WSL the NPU is queried through the Windows xrt-smi.exe.
    XRTSMI="/mnt/c/Windows/System32/AMD/xrt-smi.exe"
    NPU="${NPU:-$("$XRTSMI" examine 2>/dev/null | tr -d '\r' | grep -E "$NPUPAT" || true)}"
else
    XRTSMI="$(command -v xrt-smi 2>/dev/null || command -v xrt-smi.exe 2>/dev/null)"
    if [ -z "$XRTSMI" ] || ! test -f "$XRTSMI"; then
        echo "ERROR: xrt-smi not found. Is XRT installed (source /opt/xilinx/xrt/setup.sh)?" >&2
        return 1
    fi
    NPU=`"$XRTSMI" examine 2>/dev/null | tr -d '\r' | grep -E "$NPUPAT" || true`
fi

# Check if the current environment is NPU2
# npu4 => Strix, npu5 => Strix Halo, npu6 => Krackan
if echo "$NPU" | grep -qiE "NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[456]"; then
    export NPU2=1
else
    export NPU2=0
fi

# Ensure pyxrt is discoverable in the current Python environment.
# Legacy XRT installs put it under $XILINX_XRT/python (handled by setup.sh).
# Ubuntu packages install it to /usr/lib/python3/dist-packages/.
if ! python3 -c "import pyxrt" 2>/dev/null; then
    PYXRT_DIR=$(python3 -c "
import glob, sys, os
for p in glob.glob('/usr/lib/python3*/dist-packages/pyxrt*.so'):
    print(os.path.dirname(p)); sys.exit(0)
for p in glob.glob('/usr/lib/python3/dist-packages/pyxrt*.so'):
    print(os.path.dirname(p)); sys.exit(0)
" 2>/dev/null)
    if [ -n "$PYXRT_DIR" ]; then
        export PYTHONPATH=${PYXRT_DIR}:${PYTHONPATH}
    fi
fi

echo ""
echo "Note: Peano (llvm-aie) has not been added to PATH to avoid conflict with"
echo "      system clang/clang++. It can be found in:"
echo "      $PEANO_INSTALL_DIR/bin"
echo ""
echo "PATH              : $PATH"
echo "LD_LIBRARY_PATH   : $LD_LIBRARY_PATH"
echo "PYTHONPATH        : $PYTHONPATH"
