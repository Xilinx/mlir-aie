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

M?=512
K?=512
N?=512
m?=64
k?=64
n?=64
n_aie_cols?=4

kernels=mm_${m}x${k}x${n}
aieargs+=-m $m -k $k -n $n --n-aie-cols ${n_aie_cols}
target_suffix=${M}x${K}x${N}_${m}x${k}x${n}_${n_aie_cols}c

include ${srcdir}/../makefile-common

build/mm_${m}x${k}x${n}.o: ${kernels_dir}/mm.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBIT_WIDTH=8 -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -c $< -o ${@F}
