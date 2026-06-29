##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
targetname = single_core
hostsrc := $(firstword $(wildcard ${srcdir}/../*bfp_test.cpp))

M ?= 512
K ?= 512
N ?= 512

include ${srcdir}/../makefile_common
