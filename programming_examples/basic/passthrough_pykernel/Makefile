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

targetname = passThroughPyKernel
VPATH := ${srcdir}/../../../aie_kernels/generic
data_size = 4096
PASSTHROUGH_SIZE = ${data_size}

.PHONY: all template clean

all: build/final_${data_size}.xclbin

build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${PASSTHROUGH_SIZE} > $@

build/aie_experimental.mlir: ${srcdir}/aie2_experimental.py
	mkdir -p ${@D}
	python3 $< ${PASSTHROUGH_SIZE} > $@

mlir_diff: build/aie.mlir build/aie_experimental.mlir
	diff build/aie.mlir build/aie_experimental.mlir

build/final_${data_size}.xclbin: build/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--xclbin-name=${@F} --npu-insts-name=insts_${data_size}.txt $(<:%=../%)

build/experimental_final_${data_size}.xclbin: build/experimental_aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--xclbin-name=${@F} --npu-insts-name=experimental_insts_${data_size}.txt $(<:%=../%)

${targetname}_${data_size}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname} -DPASSTHROUGH_SIZE=${data_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}_${data_size}.exe build/final_${data_size}.xclbin build/insts_${data_size}.txt
	${powershell} ./$< -x build/final_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE 

run_py: build/final_${data_size}.xclbin build/insts_${data_size}.txt
	${powershell} python3 ${srcdir}/test.py -s ${data_size} -x build/final_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE

run_experimental: build/experimental_final_${data_size}.xclbin build/experimental_insts_${data_size}.txt
	${powershell} python3 ${srcdir}/test.py -s ${data_size} -x build/final_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE

clean:
	rm -rf build _build ${targetname}*.exe