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

include ${srcdir}/../../makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = passthrough_pykernel
VPATH := ${srcdir}/../../../aie_kernels/generic
data_size = 4096
PASSTHROUGH_SIZE = ${data_size}

aie_py_src=${targetname}.py
use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

.PHONY: all template clean

all: build/final_${data_size}.xclbin

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${PASSTHROUGH_SIZE} ${devicename} > $@

build/final_${data_size}.xclbin: build/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts_${data_size}.bin $(<:%=../%)

${targetname}_${data_size}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname} -DPASSTHROUGH_SIZE=${data_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}_${data_size}.exe build/final_${data_size}.xclbin build/insts_${data_size}.bin
	${powershell} ./$< -x build/final_${data_size}.xclbin -i build/insts_${data_size}.bin -k MLIR_AIE 

run_py: build/final_${data_size}.xclbin build/insts_${data_size}.bin
	${powershell} python3 ${srcdir}/test.py -s ${data_size} -x build/final_${data_size}.xclbin -i build/insts_${data_size}.bin -k MLIR_AIE

clean:
	rm -rf build _build ${targetname}*.exe

run_notebook: ${srcdir}/passthrough_pykernel.ipynb
	mkdir -p notebook_build
	cd notebook_build && jupyter nbconvert --to script ${srcdir}/passthrough_pykernel.ipynb --output-dir .
	cd notebook_build && ipython passthrough_pykernel.py

clean_notebook:
	rm -rf notebook_build notebook_aie.mlir
