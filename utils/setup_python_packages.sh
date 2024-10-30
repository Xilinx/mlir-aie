#!/usr/bin/env bash
##===- utils/setup_python_packages.sh - Setup python packages for mlir-aie build --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script sets up and installs the required python packages to build mlir-aie.
#
# source ./setup_python_packages.sh
#
##===----------------------------------------------------------------------===##

python3 -m venv sandbox
source sandbox/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r python/requirements.txt
HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt
