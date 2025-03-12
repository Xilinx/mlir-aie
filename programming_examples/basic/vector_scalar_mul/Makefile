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
targetname = vector_scalar_mul
# in1_size = 4096
in1_size = 8192 # in bytes
in2_size = 4 # in bytes, should always be 4 (1x int32)
out_size = 8192 # in bytes, should always be equal to in1_size
trace_size = 8192
CHESS ?= false

data_size = in1_size
aie_py_src=${targetname}.py
use_alt?=0

ifeq (${use_alt}, 1)
aie_py_src=${targetname}_alt.py
endif

all: build/final_${data_size}.xclbin build/insts_${data_size}.txt

build/%.o: %.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}; 
else 
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}; 
endif
else ifeq ($(devicename),npu2)
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}; 
else 
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}; 
endif
else
	echo "Device type not supported"
endif

build/aie_${data_size}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} ${in1_size} ${in2_size} ${out_size} 0 > $@

build/aie_trace_${data_size}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} ${in1_size} ${in2_size} ${out_size} ${trace_size} > $@

#build/insts_${data_size}.txt: build/final_${data_size}.xclbin
build/final_${data_size}.xclbin: build/aie_${data_size}.mlir build/scale.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu-insts --npu-insts-name=insts_${data_size}.txt $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    	  --no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=insts_${data_size}.txt $(<:%=../%)
endif

build/final_trace_${data_size}.xclbin: build/aie_trace_${data_size}.mlir build/scale.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu-insts --npu-insts-name=insts_${data_size}.txt $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=insts_${data_size}.txt $(<:%=../%)
endif

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

run: ${targetname}_${data_size}.exe build/final_${data_size}.xclbin build/insts_${data_size}.txt 
	${powershell} ./$< -x build/final_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE

run_py: build/final_${data_size}.xclbin build/insts_${data_size}.txt
	${powershell} python3 ${srcdir}/test.py -x build/final_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE -i1s ${in1_size} -i2s ${in2_size} -os ${out_size}

trace: ${targetname}_${data_size}.exe build/final_trace_${data_size}.xclbin build/insts_${data_size}.txt 
	${powershell} ./$< -x build/final_trace_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../utils/parse_trace.py --filename trace.txt --mlir build/aie_trace_${data_size}.mlir --colshift 1 > trace_vector_scalar_mul.json
	${srcdir}/../../utils/get_trace_summary.py --filename trace_vector_scalar_mul.json

trace_py: build/final_trace_${data_size}.xclbin build/insts_${data_size}.txt
	#${powershell} python3 ${srcdir}/test_orig.py -x build/final_trace_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE -t ${trace_size} -i1s ${in1_size} -i2s ${in2_size} -os ${out_size} --size 4096
	${powershell} python3 ${srcdir}/test.py -x build/final_trace_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE -t ${trace_size} -i1s ${in1_size} -i2s ${in2_size} -os ${out_size}
	${srcdir}/../../utils/parse_trace.py --filename trace.txt --mlir build/aie_trace_${data_size}.mlir --colshift 1 > trace_vector_scalar_mul.json
	${srcdir}/../../utils/get_trace_summary.py --filename trace_vector_scalar_mul.json


clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean: clean_trace
	rm -rf build _build ${targetname}*.exe
