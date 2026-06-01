##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir = ${srcdir}
targetname = n32_core
devicename ?= npu2
hostsrc := ${srcdir}/../gemm_atb_bfp_test.cpp

# M, K, N: final GEMM dimensions. K=4096 is also assumed by this
# config's microkernel (see the K_Problemsize constant in
# mm_bfp_mixed.cc), so this design is not K-parametric — only the
# paper K is supported.
M?=4096
K?=4096
N?=2048

# m, k, n: per-core L1 output tile (m x n) and reduction step (k).
# Asymmetry rho = 4 is encoded inside n32_core.py as DIV=4, so the L1
# A buffer is (m/4, k/8) v8bfp16ebs8 entries. Baked into the in-tree
# microkernel's hand-tuned schedule, not overridable.
m=128
k=64
n=128

include ${srcdir}/../makefile_common
