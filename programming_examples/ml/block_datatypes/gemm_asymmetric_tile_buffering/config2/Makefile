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
devicename = npu2
kernelsrc := mm_bfp_mixed.cc
hostsrc := ${srcdir}/../gemm_atb_bfp_test.cpp

kernel_dir := ${srcdir}
buffer_aloc_flag := basic-sequential

# M, K, N: final GEMM dimensions. Defaults match the paper-scale shape we
# report in the umbrella README's Performance table. The runtime sequence
# uses a 4-way ping-pong scheme, so (M/m)*(N/n) must be a multiple of 128
# if you override these.
M?=3072
K?=4096
N?=1536

# m, k, n: per-core L1 output tile (m_C x n) and reduction step (k).
# Asymmetry rho = 6 is encoded inside n32_core_placed.py as `DIV = 6`,
# so the L1 A buffer is (m/6, k/8) v8bfp16ebs8 entries. These are baked
# into the in-tree microkernel's hand-tuned schedule, so they are fixed
# (not overridable).
m=192
k=128
n=96

include ${srcdir}/../makefile_common

KERNEL_DEFINES += -I${srcdir}/../../../../../aie_kernels
