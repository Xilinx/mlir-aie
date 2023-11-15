##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../makefile-common

VPATH := ../vision_kernels

all: build/final.xclbin

build/threshold.o: threshold.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc -d ${CHESSCC2_FLAGS} -DBIT_WIDTH=8 -c $(<:%=../%) -o ${@F}
	
#build/final_%.xclbin: aie_%.mlir build/threshold.o
build/final.xclbin: aie.mlir build/threshold.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

colorThreshold.exe: test.cpp 
	rm -rf _build
	mkdir _build
	cd _build && powershell.exe cmake ../ 
	cd _build && powershell.exe cmake --build . --config Release
	cp _build/Release/colorThreshold.exe ./

run: colorThreshold.exe build/final.xclbin
	powershell.exe ./colorThreshold.exe -x ./build/final.xclbin -i ./build/insts.txt -k MLIR_AIE

clean:
	rm -rf build _build colorThreshold.exe

