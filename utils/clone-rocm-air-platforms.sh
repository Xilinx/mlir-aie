#!/usr/bin/env bash

##===- utils/clone-rocm-air-platforms.sh ---------------------*- Script -*-===##
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

##===----------------------------------------------------------------------===##
#
# This script checks out the ROCm-air-platforms repository which contains the
# driver, hardware, and firmware of the AIR ROCm platform.
#
##===----------------------------------------------------------------------===##

git clone https://github.com/Xilinx/ROCm-air-platforms
