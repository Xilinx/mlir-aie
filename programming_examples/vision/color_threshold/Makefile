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

COLORTHRESHOLD_WIDTH  = 1920
COLORTHRESHOLD_HEIGHT = 1080

#COLORTHRESHOLD_WIDTH  = 128
#COLORTHRESHOLD_HEIGHT = 64

targetname = colorThreshold

all: build/final_${COLORTHRESHOLD_WIDTH}.xclbin

mlir: build/aie2_${COLORTHRESHOLD_WIDTH}.mlir

build/%.cc.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
	
build/aie2_${COLORTHRESHOLD_WIDTH}.mlir: ${srcdir}/aie2_colorThreshold.py
	mkdir -p ${@D}
	python3 $< ${COLORTHRESHOLD_WIDTH} ${COLORTHRESHOLD_HEIGHT} > $@

build/final_${COLORTHRESHOLD_WIDTH}.xclbin: build/aie2_${COLORTHRESHOLD_WIDTH}.mlir build/threshold.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host --basic-alloc-scheme \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.txt $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname} -DCOLORTHRESHOLD_WIDTH=${COLORTHRESHOLD_WIDTH} -DCOLORTHRESHOLD_HEIGHT=${COLORTHRESHOLD_HEIGHT} -D CMAKE_C_COMPILER=gcc-13 -D CMAKE_CXX_COMPILER=g++-13
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final_${COLORTHRESHOLD_WIDTH}.xclbin build/insts.txt
	${powershell} ./$< -x build/final_${COLORTHRESHOLD_WIDTH}.xclbin -i build/insts.txt -k MLIR_AIE

clean:
	rm -rf build _build ${targetname}.exe

