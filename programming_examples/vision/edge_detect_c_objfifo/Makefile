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

targetname = edge_detect_c_objfifo

aie_py_src=edge_detect_c_objfifo.py

ifeq ($(device),npu)
PEANO_FLAGS = ${PEANOWRAP2_FLAGS}
AIE_RUNTIME_LIB = AIE2
else ifeq ($(device),npu2)
PEANO_FLAGS = ${PEANOWRAP2P_FLAGS}
AIE_RUNTIME_LIB = AIE2P
endif

AIE_OBJECTFIFO_INC = -I ${MLIR_AIE_DIR}/aie_runtime_lib/${AIE_RUNTIME_LIB}

all: build/final_${EDGEDETECT_WIDTH}.xclbin

# Existing compute kernels (from aie_kernels/aie2)
build/%.cc.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANO_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

# Wrapper kernels (from this directory) — need objectfifo.h
build/%_wrapper.cc.o: ${srcdir}/%_wrapper.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANO_FLAGS} ${AIE_OBJECTFIFO_INC} -c $< -o ${@F}

# Combined archives for each core
build/rgba2gray_wrapper.a: build/rgba2gray_wrapper.cc.o build/rgba2gray.cc.o
	ar rvs $@ $^

build/filter2d_wrapper.a: build/filter2d_wrapper.cc.o build/filter2d.cc.o
	ar rvs $@ $^

build/threshold_wrapper.a: build/threshold_wrapper.cc.o build/threshold.cc.o
	ar rvs $@ $^

build/gray2rgba_addweighted_wrapper.a: build/gray2rgba_addweighted_wrapper.cc.o build/gray2rgba.cc.o build/addWeighted.cc.o
	ar rvs $@ $^

build/aie2_${EDGEDETECT_WIDTH}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${device} ${EDGEDETECT_WIDTH} ${EDGEDETECT_HEIGHT} > $@

WRAPPER_OBJS = build/rgba2gray_wrapper.a build/filter2d_wrapper.a build/threshold_wrapper.a build/gray2rgba_addweighted_wrapper.a

build/final_${EDGEDETECT_WIDTH}.xclbin: build/aie2_${EDGEDETECT_WIDTH}.mlir $(WRAPPER_OBJS)
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

clean:
	rm -rf build _build ${targetname}.exe
