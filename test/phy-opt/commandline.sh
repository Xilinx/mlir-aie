# RUN: bash -- %s phy-opt | FileCheck %s
# CHECK: OVERVIEW: MLIR-PHY optimizer driver
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.

PHY_OPT=$1
${PHY_OPT} --help