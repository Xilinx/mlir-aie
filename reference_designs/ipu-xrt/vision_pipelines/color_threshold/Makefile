##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../makefile-common

VPATH := ../vision_kernels

targetname = colorThreshold

all: build/final.xclbin

build/threshold.o: threshold.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -d -DBIT_WIDTH=8 -c $(<:%=../%) -o ${@F}
	
# build/aie.mlir: aie2.py
# 	mkdir -p ${@D}
# 	python3 $< > $@

build/aie.mlir: aie.mlir
	mkdir -p ${@D}
	cp $< $@

build/final.xclbin: build/aie.mlir build/threshold.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

build/${targetname}.exe: test.cpp
	mkdir -p ${@D}
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake .. -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
	cp _build/${targetname}* $@

run: build/${targetname}.exe build/final.xclbin build/insts.txt
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

clean:
	rm -rf build _build colorThreshold.exe

