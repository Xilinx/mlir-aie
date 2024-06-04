##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

subdir=matrix_vector
targetname=matrixVectorMultiplication
kernels=mv

# Currently does not accept reconfiguring size via these variables; must change
# in source at aie2.py as well as here
M=288
K=288
N=1

SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include ${SELF_DIR}../makefile-common

