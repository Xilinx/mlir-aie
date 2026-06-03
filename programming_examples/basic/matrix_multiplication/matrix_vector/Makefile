##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir=matrix_vector
targetname=matrix_vector

# Sizes are fixed here to match matrix_vector.py's vectorized mv kernel
# tile (m, k). The Python CLI accepts -M/-K but the kernel object is built
# for the (m=32, k=32) tile dimensions; changing m or k requires also
# editing matrix_vector.py.
M=288
K=288
N=1
m=32
k=32

include ${srcdir}/../makefile-common

ifeq (${use_scalar}, 1)
	aieargs+=--scalar
endif
