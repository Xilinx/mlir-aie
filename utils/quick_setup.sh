#!/usr/bin/env bash
##===- quick_setup.sh - Setup IRON for Ryzen AI dev ----------*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script is the quickest path to running the Ryzen AI reference designs.
# Please have the Vitis tools and XRT environment setup before sourcing the 
# script.
#
# source ./utils/quick_setup.sh
#
##===----------------------------------------------------------------------===##

echo "Setting up RyzenAI developement tools..."
if [[ $WSL_DISTRO_NAME == "" ]]; then
  XRTSMI=`which xrt-smi`
  if ! test -f "$XRTSMI"; then 
    echo "XRT is not installed"
    return 1
  fi
  NPU=`/opt/xilinx/xrt/bin/xrt-smi examine | grep RyzenAI`
  if [[ $NPU == *"RyzenAI"* ]]; then
    echo "Ryzen AI NPU found:"
    echo $NPU
  else
    echo "NPU not found. Is the amdxdna driver installed?"
    return 1
  fi
else
  echo "Environment is WSL"
fi
# Check if the current environment is NPU2
if echo "$NPU" | grep -qi "npu4"; then
    export NPU2=1
else
    export NPU2=0
fi
if hash python3.12; then
   echo "Using python version `python3.12 --version`"
   my_python=python3.12
elif hash python3.10; then
   echo "Using python version `python3.10 --version`"
   my_python=python3.10
else
   echo "This script requires python3.10 or python3.12"
   return 1
fi

# if an install is already present, remove it to start from a clean slate
rm -rf ironenv
rm -rf my_install
$my_python -m venv ironenv
source ironenv/bin/activate
python3 -m pip install --upgrade pip

python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels/ 
export MLIR_AIE_INSTALL_DIR="$(pip show mlir_aie | grep ^Location: | awk '{print $2}')/mlir_aie"

python3 -m pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
export PEANO_INSTALL_DIR="$(pip show llvm-aie | grep ^Location: | awk '{print $2}')/llvm-aie"

python3 -m pip install -r python/requirements.txt

# This installs the pre-commit hooks defined in .pre-commit-config.yaml
pre-commit install

HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt
python3 -m pip install -r python/requirements_ml.txt

# This creates an ipykernel (for use in notebooks) using the ironenv venv
python3 -m ipykernel install --user --name ironenv

# right now, mlir-aie install dire is generally captures in the $PYTHONPATH by the setup_env script.
# However, jupyter notebooks don't always get access to the PYTHONPATH (e.g. if they are run with
# vscode) so we save the ${MLIR_AIE_INSTALL_DIR}/python in a .pth file in the site packages dir of the
# ironenv venv; this allows the iron ipykernel to find the install dir regardless of if PYTHONPATH is
# available or not.
venv_site_packages=`python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'`
echo ${MLIR_AIE_INSTALL_DIR}/python > $venv_site_packages/mlir-aie.pth

# Setup environment and add tools to PATHs
source utils/env_setup.sh

pushd programming_examples
echo "PATH              : $PATH"
echo "LD_LIBRARY_PATH   : $LD_LIBRARY_PATH"
echo "PYTHONPATH        : $PYTHONPATH"
