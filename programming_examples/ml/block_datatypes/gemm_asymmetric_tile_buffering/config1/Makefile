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
hostsrc := ${srcdir}/../gemm_atb_mixed_test.cpp

# This example ships its own kernel inside the example directory.
# The shared makefile_common honors this via `kernel_dir ?=`.
kernel_dir := ${srcdir}

# `basic-sequential` matches what the paper measurements used.
buffer_aloc_flag := basic-sequential

# M, K, N: final GEMM dimensions. Defaults match the paper-scale shape we
# report in the umbrella README's Performance table. The runtime sequence
# uses a 4-way ping-pong scheme, so (M/m)*(N/n) must be a multiple of
# (4 * n_aie_cols * n_aie_rows) = 128 if you override these.
M?=4096
K?=4096
N?=2048

# m, k, n: per-core L1 output tile (m_C x n) and reduction step (k).
# Asymmetry rho = m_C / m_A = 4 is encoded inside n32_core_placed.py
# (A_l1 = (m/4, k), C_l1 = (m, n)).
m?=128
k?=64
n?=128

include ${srcdir}/../makefile_common

# Help the compiler find aie_kernel_utils.h, which the in-tree kernel
# includes by its bare name.
KERNEL_DEFINES += -I${srcdir}/../../../../../aie_kernels
