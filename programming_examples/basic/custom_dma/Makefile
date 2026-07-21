##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

devicename ?= npu2
targetname = custom_dma

.PHONY: all clean

all: build/final.xclbin

build/aie.mlir: ${srcdir}/${targetname}.py
	mkdir -p ${@D}
	python3 $< -d ${devicename} > $@

build/final.xclbin: build/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc --get-xclbin --get-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

run_py: build/final.xclbin
	${powershell} python3 ${srcdir}/test.py --xclbin build/final.xclbin --instr build/insts.bin -k MLIR_AIE

clean:
	rm -rf build _build
