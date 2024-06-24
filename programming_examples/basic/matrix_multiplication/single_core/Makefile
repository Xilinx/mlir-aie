##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir=single_core
targetname=matrixMultiplication
kernels=mm

M?=256
K?=256
N?=256
m?=64
k?=64
n?=64

kernels=mm_${m}x${k}x${n}
aieargs+=-m $m -k $k -n $n
target_suffix=${M}x${K}x${N}_${m}x${k}x${n}

include ${srcdir}/../makefile-common

build/mm_${m}x${k}x${n}.o: ${kernels_dir}/mm.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBIT_WIDTH=8 -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -c $< -o ${@F}

