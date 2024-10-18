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
targetname=matrixMultiplication

M?=640
K?=896
N?=768
m?=16
k?=32
n?=48
n_aie_cols?=2
b_col_maj?=0

kernels=mm_${m}x${k}x${n}
aieargs+=-m $m -k $k -n $n --n-aie-cols ${n_aie_cols} --b-col-maj ${b_col_maj}
runargs+=--b_col_maj ${b_col_maj}
target_suffix=${M}x${K}x${N}_${m}x${k}x${n}_${n_aie_cols}c

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