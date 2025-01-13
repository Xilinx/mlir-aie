##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir=whole_array
targetname=whole_array

# M, K, N: final GEMM dimensions
M?=512
K?=512
N?=512

# m, k, n: single AIE core GEMM dimensions
m?=64
k?=64
n?=64

n_aie_cols?=4
b_col_maj?=0

kernels=mm_${m}x${k}x${n}
aieargs+=-m $m -k $k -n $n --n-aie-cols ${n_aie_cols} --b-col-maj ${b_col_maj}
runargs+=--b_col_maj ${b_col_maj}
target_suffix=${M}x${K}x${N}_${m}x${k}x${n}_${n_aie_cols}c
use_alt?=0
use_iron?=0

# set this flag to 1 for linear buffer allocation
# else, 0 for bank-aware
use_linear_buf_alloc?=1

ifeq (${use_linear_buf_alloc}, 1)
buffer_aloc_flag=basic-sequential
else
buffer_aloc_flag=bank-aware
endif

ifeq (${use_alt}, 1)
aie_py_src=${targetname}_alt.py
ifeq (${use_iron}, 1)
$(error Cannot specify both alt design and IRON)
endif
endif

ifeq (${use_iron}, 1)
aie_py_src=${targetname}_iron.py
endif

include ${srcdir}/../makefile-common

build/mm_b_row_maj_${m}x${k}x${n}.o: ${kernels_dir}/mm.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -c $< -o ${@F}

build/mm_b_col_maj_${m}x${k}x${n}.o: ${kernels_dir}/mm_b_col_maj.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -c $< -o ${@F}

ifneq (${b_col_maj}, 1)
build/mm_${m}x${k}x${n}.o: build/mm_b_row_maj_${m}x${k}x${n}.o 
	mkdir -p ${@D}
	ar rvs $@ $^
else
build/mm_${m}x${k}x${n}.o: build/mm_b_col_maj_${m}x${k}x${n}.o
	mkdir -p ${@D}
	ar rvs $@ $^
endif