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

include ${srcdir}/../../../programming_examples/makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = aie2

in1_size = 16384 # in bytes
in2_size = 4 # in bytes, should always be 4 (1x int32)
out_size = 16384 # in bytes, should always be equal to in1_size
trace_size = 8192

colshift ?= $(if $(filter npu,$(devicename)),1,0)
aie_py_src=${targetname}.py

use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

all: build/final.xclbin build/insts.bin

build/scale.o: ${srcdir}/vector_scalar_mul.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}; 
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}; 
else
	echo "Device type not supported"
endif

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -i1s ${in1_size} -i2s ${in2_size} -os ${out_size} > $@

build/aie_trace.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -i1s ${in1_size} -i2s ${in2_size} -os ${out_size} -t ${trace_size} > $@

build/final.xclbin: build/aie.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
			--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

build/final_trace.xclbin: build/aie_trace.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
			--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname} -DIN1_SIZE=${in1_size} -DIN2_SIZE=${in2_size} -DOUT_SIZE=${out_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin build/insts.bin 
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE

run_py: build/final.xclbin build/insts.bin
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -i1s ${in1_size} -i2s ${in2_size} -os ${out_size}

trace: ${targetname}.exe build/final_trace.xclbin build/insts.bin 
	${powershell} ./$< -x build/final_trace.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../programming_examples/utils/parse_trace.py --filename trace.txt --mlir build/aie_trace.mlir --colshift ${colshift} > trace_4b.json
	${srcdir}/../../../programming_examples/utils/get_trace_summary.py --filename trace_4b.json

trace_py: build/final_trace.xclbin build/insts.bin
	${powershell} python3 ${srcdir}/test.py -x build/final_trace.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size} -i1s ${in1_size} -i2s ${in2_size} -os ${out_size}
	${srcdir}/../../../programming_examples/utils/parse_trace.py --filename trace.txt --mlir build/aie_trace.mlir --colshift ${colshift} > trace_4b.json
	${srcdir}/../../../programming_examples/utils/get_trace_summary.py --filename trace_4b.json


clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean: clean_trace
	rm -rf build _build ${targetname}*.exe
