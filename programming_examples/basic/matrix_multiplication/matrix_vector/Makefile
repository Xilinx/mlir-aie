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

# Currently does not accept reconfiguring size via these variables; must change
# in source at <targetname>.py as well as here.
M=288
K=288
N=1
m=32
k=32

include ${srcdir}/../makefile-common

ifeq (${use_scalar}, 1)
	aieargs+=--scalar
endif
