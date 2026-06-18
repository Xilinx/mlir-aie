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
subdir=whole_array
targetname=whole_array

M?=512
K?=512
N?=512

m?=32
k?=32
n?=32

n_aie_cols?=4
b_col_maj?=0
c_col_maj?=0
use_scalar?=0

aieargs+=-m $m -k $k -n $n --n-aie-cols ${n_aie_cols} --b-col-maj ${b_col_maj} --c-col-maj ${c_col_maj}
ifeq (${use_scalar}, 1)
	aieargs+= --scalar 1
endif
runargs+=--b_col_maj ${b_col_maj} --c_col_maj ${c_col_maj}
target_suffix=${M}x${K}x${N}_${m}x${k}x${n}_${n_aie_cols}c

include ${srcdir}/../makefile-common
