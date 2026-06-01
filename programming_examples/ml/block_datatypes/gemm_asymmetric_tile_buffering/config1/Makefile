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
hostsrc := ${srcdir}/../gemm_atb_mixed_test.cpp

# M, K, N: final GEMM dimensions. The runtime sequence uses a 4-deep DMA
# pipeline, so (M/m)*(N/n) must be a multiple of
# (4 * n_aie_cols * n_aie_rows) = 128 when overriding.
M?=4096
K?=4096
N?=2048

# m, k, n: per-core L1 output tile (m x n) and reduction step (k).
# Asymmetry rho = m_C / m_A = 4 is encoded inside n32_core.py (A_l1 =
# (m/4, k), C_l1 = (m, n)). These are baked into the in-tree microkernel
# and are not overridable.
m=128
k=64
n=128

include ${srcdir}/../makefile_common
