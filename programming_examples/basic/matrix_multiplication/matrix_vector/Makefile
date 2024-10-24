##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

subdir=matrix_vector
targetname=matrixVectorMultiplication

# Currently does not accept reconfiguring size via these variables; must change
# in source at aie2.py as well as here
M=288
K=288
N=1
m=32
k=32

kernels=mv_${m}x${k}
use_experimental?=0

ifeq (${use_experimental}, 1)
aie_py_src=experimental_aie2.py
endif

SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include ${SELF_DIR}../makefile-common

build/mv_${m}x${k}.o: ${kernels_dir}/mv.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -DDIM_M=${m} -DDIM_K=${k} -c $< -o ${@F}

