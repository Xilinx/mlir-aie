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
targetname = eltwise_add
in1_size = 131072 # in bytes
in2_size = 131072 # in bytes, should always be euqal to in1_size
out_size = 131072 # in bytes, should always be equal to in1_size
trace_size = 8192
CHESS ?= false

colshift ?= $(if $(filter npu,$(devicename)),1,0)
data_size = in1_size
aie_py_src=${targetname}.py
use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

all: build/final_${data_size}.xclbin build/insts_${data_size}.bin

build/%.o: %.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}; 
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}; 
else
	echo "Device type not supported"
endif

build/aie_${data_size}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -i1s ${in1_size} -i2s ${in2_size} -os ${out_size} > $@

build/aie_trace_${data_size}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -i1s ${in1_size} -i2s ${in2_size} -os ${out_size} -t ${trace_size} > $@

build/final_${data_size}.xclbin: build/aie_${data_size}.mlir build/add.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    	  --no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=insts_${data_size}.bin $(<:%=../%)

build/final_trace_${data_size}.xclbin: build/aie_trace_${data_size}.mlir build/add.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=insts_${data_size}.bin $(<:%=../%)

${targetname}_${data_size}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname}_${data_size} -DIN1_SIZE=${in1_size} -DIN2_SIZE=${in2_size} -DOUT_SIZE=${out_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}_${data_size}.exe $@
else
	cp _build/${targetname}_${data_size} $@ 
endif

run: ${targetname}_${data_size}.exe build/final_${data_size}.xclbin build/insts_${data_size}.bin 
	${powershell} ./$< -x build/final_${data_size}.xclbin -i build/insts_${data_size}.bin -k MLIR_AIE

trace: ${targetname}_${data_size}.exe build/final_trace_${data_size}.xclbin build/insts_${data_size}.bin 
	${powershell} ./$< -x build/final_trace_${data_size}.xclbin -i build/insts_${data_size}.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../utils/parse_trace.py --filename trace.txt --mlir build/aie_trace_${data_size}.mlir --colshift ${colshift} > trace_${targetname}.json
	${srcdir}/../../utils/get_trace_summary.py --filename trace_${targetname}.json

trace_py: build/final_trace_${data_size}.xclbin build/insts_${data_size}.bin
	${powershell} python3 ${srcdir}/test.py -x build/final_trace_${data_size}.xclbin -i build/insts_${data_size}.bin -k MLIR_AIE -t ${trace_size} -i1s ${in1_size} -i2s ${in2_size} -os ${out_size}
	${srcdir}/../../utils/parse_trace.py --filename trace.txt --mlir build/aie_trace_${data_size}.mlir --colshift ${colshift} > trace_${targetname}.json
	${srcdir}/../../utils/get_trace_summary.py --filename trace_${targetname}.json

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean: clean_trace
	rm -rf build _build ${targetname}*.exe
