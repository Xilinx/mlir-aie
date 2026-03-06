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
COLORTHRESHOLD_WIDTH  = 1920
COLORTHRESHOLD_HEIGHT = 1080

#COLORTHRESHOLD_WIDTH  = 128
#COLORTHRESHOLD_HEIGHT = 64

targetname = color_threshold

aie_py_src=${targetname}.py
use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

all: build/final_${COLORTHRESHOLD_WIDTH}.xclbin

mlir: build/aie2_${COLORTHRESHOLD_WIDTH}.mlir

build/%.cc.o: %.cc
	mkdir -p ${@D}
ifeq ($(device),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else ifeq ($(device),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else
	echo "Device type not supported"
endif
	
build/aie2_${COLORTHRESHOLD_WIDTH}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${device} ${COLORTHRESHOLD_WIDTH} ${COLORTHRESHOLD_HEIGHT} > $@

build/final_${COLORTHRESHOLD_WIDTH}.xclbin: build/aie2_${COLORTHRESHOLD_WIDTH}.mlir build/threshold.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname} -DCOLORTHRESHOLD_WIDTH=${COLORTHRESHOLD_WIDTH} -DCOLORTHRESHOLD_HEIGHT=${COLORTHRESHOLD_HEIGHT} -D CMAKE_C_COMPILER=${CC} -D CMAKE_CXX_COMPILER=${CXX}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final_${COLORTHRESHOLD_WIDTH}.xclbin
	${powershell} ./$< -x build/final_${COLORTHRESHOLD_WIDTH}.xclbin -i build/insts.bin -k MLIR_AIE

clean:
	rm -rf build _build ${targetname}.exe
