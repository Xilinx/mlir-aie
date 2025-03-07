##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../../programming_examples/makefile-common

all: build/final.xclbin build/insts.txt

targetname = single_buffer

aie_py_src=${targetname}.py
use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< > $@

.PHONY: inst/insts.txt
inst/insts.txt: ${srcdir}/${aie_py_src}
	rm -rf inst
	mkdir -p inst 
	python3 $< ${LENGTH} > inst/aie.mlir
	pushd inst && aiecc.py --aie-only-generate-npu --npu-insts-name=insts.txt aie.mlir && popd
	${powershell} ./build/${targetname}.exe -x build/final.xclbin -i inst/insts.txt -k MLIR_AIE -l ${LENGTH}

build/final.xclbin: build/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=insts.txt $(<:%=../%)

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
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

run_py: build/final.xclbin build/insts.txt
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

clean: 
	rm -rf build _build ${targetname}.exe

