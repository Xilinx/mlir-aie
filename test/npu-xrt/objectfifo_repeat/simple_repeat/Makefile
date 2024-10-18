##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

all: build/final.xclbin build/insts.txt

devicename ?= npu
targetname = simple_repeat
LENGTH ?= 4096

build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${LENGTH} ${devicename} ${col} > $@

.PHONY: inst/insts.txt
inst/insts.txt: ${srcdir}/aie2.py
	rm -rf inst
	mkdir -p inst
	python3 $< ${LENGTH} > inst/aie.mlir
	pushd inst && aiecc.py --aie-only-generate-npu --npu-insts-name=insts.txt aie.mlir && popd
	${powershell} ./build/${targetname}.exe -x build/final.xclbin -i inst/insts.txt -k MLIR_AIE -l ${LENGTH}

build/final.xclbin: build/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
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

run: ${targetname}.exe build/final.xclbin build/insts.txt
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE -l ${LENGTH}

clean:
	rm -rf build _build inst ${targetname}.exe
