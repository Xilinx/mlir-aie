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

# M, K, N: final GEMM dimensions. The runtime sequence uses a 4-deep DMA
# pipeline, so (M/m)*(N/n) must be a multiple of 128 when overriding.
M?=3072
K?=4096
N?=1536

# m, k, n: per-core L1 output tile (m x n) and reduction step (k).
# Asymmetry rho = 6 is encoded inside n32_core.py as DIV=6, so the L1 A
# buffer is (m/6, k/8) v8bfp16ebs8 entries. Baked into the in-tree
# microkernel's hand-tuned schedule, not overridable.
m=192
k=128
n=96

include ${srcdir}/../makefile_common
