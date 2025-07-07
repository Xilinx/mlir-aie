##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

VPATH := ${srcdir}/../../../aie_kernels/aie2

device ?= $(if $(filter 1,$(NPU2)),npu2,npu)
COLORDETECT_WIDTH = 1920
COLORDETECT_HEIGHT = 1080

targetname = color_detect

#COLORDETECT_WIDTH = 64
#COLORDETECT_HEIGHT = 36

#COLORDETECT_WIDTH = 640
#COLORDETECT_HEIGHT = 480

aie_py_src=${targetname}.py
use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

all: build/final_${COLORDETECT_WIDTH}.xclbin

mlir: build/aie2_lineBased_8b_${COLORDETECT_WIDTH}.mlir

build/%.cc.o: %.cc
	mkdir -p ${@D}
ifeq ($(device),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else ifeq ($(device),npu2)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else
	echo "Device type not supported"
endif

build/combined_bitwiseOR_gray2rgba_bitwiseAND.a: build/bitwiseOR.cc.o build/gray2rgba.cc.o build/bitwiseAND.cc.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

build/aie2_lineBased_8b_${COLORDETECT_WIDTH}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${device} ${COLORDETECT_WIDTH} ${COLORDETECT_HEIGHT} > $@

build/final_${COLORDETECT_WIDTH}.xclbin: build/aie2_lineBased_8b_${COLORDETECT_WIDTH}.mlir build/rgba2hue.cc.o build/threshold.cc.o build/combined_bitwiseOR_gray2rgba_bitwiseAND.a
	mkdir -p ${@D}
ifeq ($(device),npu)
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)
endif

${targetname}.exe: ${srcdir}/test.cpp
	mkdir -p ${@D}
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname} -DCOLORDETECT_WIDTH=${COLORDETECT_WIDTH} -DCOLORDETECT_HEIGHT=${COLORDETECT_HEIGHT} -D CMAKE_C_COMPILER=gcc-13 -D CMAKE_CXX_COMPILER=g++-13
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final_${COLORDETECT_WIDTH}.xclbin
	${powershell} ./$< -x build/final_${COLORDETECT_WIDTH}.xclbin -i build/insts.bin -k MLIR_AIE

run_live: ${targetname}.exe build/final_${COLORDETECT_WIDTH}.xclbin
	${powershell} ./$< -l -x build/final_${COLORDETECT_WIDTH}.xclbin -i build/insts.bin -k MLIR_AIE

clean:
	rm -rf build _build ${targetname}.exe colorDetectOut_test.jpg
