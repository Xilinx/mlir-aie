##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

subdir=single_core
targetname=matrixMultiplication
kernels=mm

# Currently does not accept reconfiguring size via these variables; must change
# in source at aie2.py as well as here
M=256
K=256
N=256

SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include ${SELF_DIR}../makefile-common

