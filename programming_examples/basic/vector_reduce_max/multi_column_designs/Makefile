##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common

VPATH := ${srcdir}/../../../../aie_kernels/aie2

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = vector_reduce_max
in1_size =  524288# in bytes
out_size = 4 # in bytes
trace_size = 524288

dtype?=i32

use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=row_wise_${targetname}_placed.py
else
aie_py_src=${targetname}.py
endif

ifeq ($(dtype),bf16)
	dtype_cpp=std::bfloat16_t
endif
ifeq ($(dtype),i32)
	dtype_cpp=std::int32_t
endif

aieargs+=--dtype ${dtype}

.PHONY: all template clean

all: build/final.xclbin

build/%.cc.o: %.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
endif

build/aie2.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -i1s ${in1_size} -os ${out_size} ${aieargs} > $@

build/aie2_trace.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -i1s ${in1_size} -os ${out_size} -t ${trace_size} ${aieargs} > $@

build/final.xclbin: build/aie2.mlir build/reduce_max.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

build/final_trace.xclbin: build/aie2_trace.mlir build/reduce_max.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)
${targetname}.exe: ${srcdir}/../test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}/..` -DTARGET_NAME=${targetname}  -DIN1_SIZE=${in1_size} -DOUT_SIZE=${out_size} -DDTYPE=${dtype_cpp} 
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin build/insts.bin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE 

run_py: build/final.xclbin build/insts.bin
	${powershell} python3 ${srcdir}/../test.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -i1s ${in1_size} -os ${out_size}

trace: ${targetname}.exe build/final_trace.xclbin build/insts.bin 
	${powershell} ./$< -x build/final_trace.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../../python/utils/parse_trace.py --input trace.txt --mlir build/aie2_trace.mlir --output trace_vector_reduce_max.json
	${srcdir}/../../../../python/utils/get_trace_summary.py --input trace_vector_reduce_max.json

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean:
	rm -rf build _build ${targetname}*.exe