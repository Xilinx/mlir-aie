##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../makefile-common

VPATH := ../vision_kernels

EDGEDETECT_WIDTH = 1920
EDGEDETECT_HEIGHT = 1080

#EDGEDETECT_WIDTH = 640
#EDGEDETECT_HEIGHT = 480

#EDGEDETECT_WIDTH = 64
#EDGEDETECT_HEIGHT = 36

targetname = edgeDetect

all: build/final_${EDGEDETECT_WIDTH}.xclbin

mlir: build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir

build/%.cc.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBIT_WIDTH=8 -c $(<:%=../%) -o ${@F}

build/combined_gray2rgba_addWeighted.a: build/gray2rgba.cc.o build/addWeighted.cc.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^)

build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir: aie2_edgeDetect.py
	mkdir -p ${@D}
	python3 $< ${EDGEDETECT_WIDTH} ${EDGEDETECT_HEIGHT} > $@

build/final_${EDGEDETECT_WIDTH}.xclbin: build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir build/rgba2gray.cc.o build/gray2rgba.cc.o build/filter2d.cc.o build/threshold.cc.o build/addWeighted.cc.o build/combined_gray2rgba_addWeighted.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

${targetname}.exe: test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake .. -DTARGET_NAME=${targetname} -DEDGEDETECT_WIDTH=${EDGEDETECT_WIDTH} -DEDGEDETECT_HEIGHT=${EDGEDETECT_HEIGHT} -D CMAKE_C_COMPILER=gcc-13 -D CMAKE_CXX_COMPILER=g++-13
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final_${EDGEDETECT_WIDTH}.xclbin build/insts.txt
	${powershell} ./$< -x build/final_${EDGEDETECT_WIDTH}.xclbin -i build/insts.txt -k MLIR_AIE

clean:
	rm -rf build _build ${targetname}.exe
