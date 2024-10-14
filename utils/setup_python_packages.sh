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

python3 -m virtualenv sandbox
# The real path to source might depend on the virtualenv version
if [ -r sandbox/local/bin/activate ]; then
  source sandbox/local/bin/activate
else
  source sandbox/bin/activate
fi
python3 -m pip install --upgrade pip
python3 -m pip install -r python/requirements.txt
HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt
