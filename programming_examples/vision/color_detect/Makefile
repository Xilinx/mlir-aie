##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../makefile-common

VPATH := ../../../aie_kernels/aie2

COLORDETECT_WIDTH = 1920
COLORDETECT_HEIGHT = 1080

targetname = colorDetect

#COLORDETECT_WIDTH = 64
#COLORDETECT_HEIGHT = 36

#COLORDETECT_WIDTH = 640
#COLORDETECT_HEIGHT = 480

all: build/final_${COLORDETECT_WIDTH}.xclbin

mlir: build/aie2_lineBased_8b_${COLORDETECT_WIDTH}.mlir

build/%.cc.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBIT_WIDTH=8 -c $(<:%=../%) -o ${@F}

build/combined_bitwiseOR_gray2rgba_bitwiseAND.a: build/bitwiseOR.cc.o build/gray2rgba.cc.o build/bitwiseAND.cc.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

build/aie2_lineBased_8b_${COLORDETECT_WIDTH}.mlir: aie2_colorDetect.py
	mkdir -p ${@D}
	python3 $< ${COLORDETECT_WIDTH} ${COLORDETECT_HEIGHT} > $@

build/final_${COLORDETECT_WIDTH}.xclbin: build/aie2_lineBased_8b_${COLORDETECT_WIDTH}.mlir build/rgba2hue.cc.o build/threshold.cc.o build/combined_bitwiseOR_gray2rgba_bitwiseAND.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--xclbin-name=${@F} --npu-insts-name=insts.txt $(<:%=../%)

build/${targetname}.exe: test.cpp
	mkdir -p ${@D}
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake .. -DTARGET_NAME=${targetname} -DCOLORDETECT_WIDTH=${COLORDETECT_WIDTH} -DCOLORDETECT_HEIGHT=${COLORDETECT_HEIGHT} -D CMAKE_C_COMPILER=gcc-13 -D CMAKE_CXX_COMPILER=g++-13

	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: build/${targetname}.exe build/final_${COLORDETECT_WIDTH}.xclbin build/insts.txt
	${powershell} ./$< -x build/final_${COLORDETECT_WIDTH}.xclbin -i build/insts.txt -k MLIR_AIE

clean:
	rm -rf build _build
