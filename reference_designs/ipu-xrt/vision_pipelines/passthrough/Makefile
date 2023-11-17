##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../makefile-common

VPATH := ../vision_kernels

WIDTH = 1920
HEIGHT = 1080

targetname = passThrough

.PHONY: all template clean

all: build/final_${WIDTH}.xclbin

build/aie2_lineBased_8b_${WIDTH}.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< ${WIDTH} ${HEIGHT} > $@

build/passThrough.cc.o: passThrough.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper -d ${CHESSCCWRAP2_FLAGS} -DBIT_WIDTH=8 -c $(<:%=../%) -o ${@F}
	
build/final_${WIDTH}.xclbin: build/aie2_lineBased_8b_${WIDTH}.mlir build/passThrough.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

build/${targetname}.exe: test.cpp
	mkdir -p ${@D}
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake .. -DTARGET_NAME=${targetname} -DWIDTH=${WIDTH} -DHEIGHT=${HEIGHT} 
	cd _build && ${powershell} cmake --build . --config Release
	cp _build/${targetname}* $@

run: build/${targetname}.exe build/final_${WIDTH}.xclbin build/insts.txt
	${powershell} ./$< -x build/final_${WIDTH}.xclbin -i build/insts.txt -k MLIR_AIE

clean:
	rm -rf build
