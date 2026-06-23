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
targetname = whole_array
hostsrc := $(firstword $(wildcard ${srcdir}/*test.cpp))

n_aie_cols ?= 4
aieargs += --n-aie-cols ${n_aie_cols}

M ?= 512
K ?= 512
N ?= 512

include ${srcdir}/../makefile_common
