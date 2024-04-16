##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../makefile-common

ACDC_AIE = $(dir $(shell which aie-opt))/..

SHELL := /bin/bash

all: build/final.xclbin build/insts.txt

targetname = passThroughDMAs
LENGTH ?= 4096

build/aie.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< ${devicename} ${col} ${LENGTH} ${LENGTH} > $@

.PHONY: inst/insts.txt
inst/insts.txt: aie2.py
	rm -rf inst
	mkdir -p inst 
	python3 $< ${devicename} ${col} ${LENGTH} > inst/aie.mlir
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
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE -l ${LENGTH}

# Changing variables when we target VCK5000
vck5000: devicename=xcvc1902
vck5000: col=6

vck5000: build/aie.mlir
	aiecc.py --link_against_hsa --host-target=x86_64-amd-linux-gnu build/aie.mlir \
		-I/opt/xaiengine/include \
		-I$(ACDC_AIE)/runtime_lib/x86_64-hsa/test_lib/include \
		-I${ROCM_ROOT}/include \
		-L/opt/xaiengine/lib \
		-L/lib/x86_64-linux-gnu/ \
		test_vck5000.cpp \
		$(ACDC_AIE)/runtime_lib/x86_64-hsa/test_lib/src/test_library.cpp \
		${ROCM_ROOT}/lib/libhsa-runtime64.so.1.9.0 \
		-Wl,-R/opt/xaiengine/lib \
		-Wl,-rpath,${ROCM_ROOT}/lib \
		-Wl,--whole-archive -Wl,--no-whole-archive -lstdc++ -ldl -lelf -o test.elf



clean:
	rm -rf build _build inst ${targetname}.exe
