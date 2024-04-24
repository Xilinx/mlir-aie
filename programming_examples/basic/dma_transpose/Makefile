##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../makefile-common

SHELL := /bin/bash

all: build/final.xclbin build/insts.txt

targetname = dmaTranspose
M ?= 64
K ?= 32

build/aie.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< ${M} ${K} > $@

.PHONY: inst/insts.txt
inst/insts.txt: aie2.py
	rm -rf inst
	mkdir -p inst 
	python3 $< ${LENGTH} > inst/aie.mlir
	pushd inst && aiecc.py --aie-only-generate-ipu --ipu-insts-name=insts.txt aie.mlir && popd
	${powershell} ./build/${targetname}.exe -x build/final.xclbin -i inst/insts.txt -k MLIR_AIE -l ${LENGTH}

build/final.xclbin: build/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-ipu --ipu-insts-name=insts.txt $(<:%=../%)

${targetname}.exe: test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake .. -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin build/insts.txt
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE --M ${M} --K ${K}

clean:
	rm -rf build _build inst ${targetname}.exe
