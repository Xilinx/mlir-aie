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
EDGEDETECT_WIDTH = 1920
EDGEDETECT_HEIGHT = 1080

#EDGEDETECT_WIDTH = 640
#EDGEDETECT_HEIGHT = 480

#EDGEDETECT_WIDTH = 64
#EDGEDETECT_HEIGHT = 36

targetname = edge_detect

aie_py_src=${targetname}.py
use_placed?=0

webcam ?= 0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

all: build/final_${EDGEDETECT_WIDTH}.xclbin

mlir: build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir

build/%.cc.o: %.cc
	mkdir -p ${@D}
ifeq ($(device),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else ifeq ($(device),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else
	echo "Device type not supported"
endif

build/combined_gray2rgba_addWeighted.a: build/gray2rgba.cc.o build/addWeighted.cc.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^)

build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${device} ${EDGEDETECT_WIDTH} ${EDGEDETECT_HEIGHT} > $@

build/final_${EDGEDETECT_WIDTH}.xclbin: build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir build/rgba2gray.cc.o build/gray2rgba.cc.o build/filter2d.cc.o build/threshold.cc.o build/addWeighted.cc.o build/combined_gray2rgba_addWeighted.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname} -DEDGEDETECT_WIDTH=${EDGEDETECT_WIDTH} -DEDGEDETECT_HEIGHT=${EDGEDETECT_HEIGHT} -D CMAKE_C_COMPILER=${CC} -D CMAKE_CXX_COMPILER=${CXX}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final_${EDGEDETECT_WIDTH}.xclbin
	${powershell} ./$< -x build/final_${EDGEDETECT_WIDTH}.xclbin -i build/insts.bin -k MLIR_AIE

run_live: ${targetname}.exe build/final_${EDGEDETECT_WIDTH}.xclbin
	${powershell} ./$< -l -x build/final_${EDGEDETECT_WIDTH}.xclbin -i build/insts.bin -k MLIR_AIE -d ${webcam}
clean:
	rm -rf build _build ${targetname}.exe ${targetname}Out_test.jpg
