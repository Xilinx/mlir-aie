##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

subdir=matrix_vector
targetname=matrix_vector

# Currently does not accept reconfiguring size via these variables; must change
# in source at <targetname>.py as well as here
M=288
K=288
N=1
m=32
k=32

kernels=mv_${m}x${k}
use_placed?=0
use_iron?=0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
ifeq (${use_iron}, 1)
$(error Cannot specify both placed design and IRON)
endif
endif

ifeq (${use_iron}, 1)
aie_py_src=${targetname}_iron.py
endif

KERNEL_DEFINES=-DDIM_M=${m} -DDIM_K=${k}
ifeq (${b_col_maj}, 1)
	KERNEL_DEFINES+=-DB_COL_MAJ
endif

SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include ${SELF_DIR}../makefile-common

# Overwrite the kernels_dir defined in makefile-common for this example; on 
# AIE2p, we will also use the AIE2 kernel -- there is no specialized Strix 
# variant.
kernels_dir=${srcdir}/../../../../aie_kernels/aie2

build/mv_${m}x${k}.o: ${kernels_dir}/mv.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c $< -o ${@F}
