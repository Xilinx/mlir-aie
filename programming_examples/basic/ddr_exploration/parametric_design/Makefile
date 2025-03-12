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

include ${srcdir}/../../../makefile-common

SHELL := /bin/bash

all: build/final.xclbin build/insts.txt

targetname = DDR_bandwidth_test
m ?= 32
M ?= 4096
k ?= 128
K ?= 4096
n ?= 32
N ?= 4096

npu_cols ?= 4


# set this flag to 1 for both row-major
# else, 0 for B in col-major
both_row?=0

ifeq (${both_row}, 1)
data_layout_DDR=A_row_B_row
else
data_layout_DDR=A_row_B_col
endif



aie_py_src=${targetname}.py


build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${m} ${M} ${k} ${K} ${n} ${N} ${npu_cols} ${data_layout_DDR} > $@

build/final.xclbin: build/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --alloc-scheme=basic-sequential --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge \
				--aie-generate-npu --npu-insts-name=insts.txt $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE --m ${m} --M ${M} --k ${k} --K ${K} --n ${n} --N ${N}

clean:
	rm -rf build _build inst ${targetname}.exe
