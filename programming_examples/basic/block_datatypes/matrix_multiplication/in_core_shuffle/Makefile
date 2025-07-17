##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir=${srcdir}
targetname=single_core
devicename=npu2
kernelsrc:= $(firstword $(wildcard ../*mm.cc))
hostsrc := $(firstword $(wildcard *.cpp))

# M, K, N: final GEMM dimensions
M?=64
K?=64
N?=64

# m, k, n: single AIE core GEMM dimensions
m?=64
k?=64
n?=64

include ${srcdir}/../makefile_common
