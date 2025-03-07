##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

VPATH := ${srcdir}/../../../aie_kernels/aie2

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = vector_reduce_max
in1_size = 4096 # in bytes
out_size = 4 # in bytes, should always be equal to in1_size
trace_size = 1024


aie_py_src=${targetname}.py
use_alt?=0

ifeq (${use_alt}, 1)
aie_py_src=${targetname}_alt.py
endif

.PHONY: all template clean

all: build/final.xclbin

build/aie2.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} ${in1_size} ${out_size} 0 > $@

build/aie2_trace.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} ${in1_size} ${out_size} ${trace_size} > $@

build/%.cc.o: %.cc
	mkdir -p ${@D}

ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
endif

build/final.xclbin: build/aie2.mlir build/reduce_max.cc.o
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.txt $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--xclbin-name=${@F} --npu-insts-name=insts.txt $(<:%=../%)
endif

build/final_trace.xclbin: build/aie2_trace.mlir build/reduce_max.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    			--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				--aie-generate-npu-insts --npu-insts-name=insts.txt $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname}  -DIN1_SIZE=${in1_size} -DOUT_SIZE=${out_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin build/insts.txt
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE 

run_py: build/final.xclbin build/insts.txt
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.txt -k MLIR_AIE -i1s ${in1_size} -os ${out_size}

trace: ${targetname}.exe build/final_trace.xclbin build/insts.txt 
	${powershell} ./$< -x build/final_trace.xclbin -i build/insts.txt -k MLIR_AIE -t ${trace_size} -v 1
	${srcdir}/../../utils/parse_trace.py --filename trace.txt --mlir build/aie2_trace.mlir --colshift 1 > trace_vector_reduce_max.json
	${srcdir}/../../utils/get_trace_summary.py --filename trace_vector_reduce_max.json

trace_py: build/final_trace.xclbin build/insts.txt
	${powershell} python3 ${srcdir}/test.py -x build/final_trace.xclbin -i build/insts.txt -k MLIR_AIE -t ${trace_size} -i1s ${in1_size} -os ${out_size}
	${srcdir}/../../utils/parse_trace.py --filename trace.txt --mlir build/aie2_trace.mlir --colshift 1 > trace_vector_reduce_max.json
	${srcdir}/../../utils/get_trace_summary.py --filename trace_vector_reduce_max.json

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean:
	rm -rf build _build ${targetname}*.exe
