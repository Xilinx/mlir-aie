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
VPATH := ${srcdir}/../../../aie_kernels/generic

devicename ?= $(if $(NPU2),npu2,npu)
targetname = passthrough_kernel
in1_size = 4096 # in bytes
out_size = 4096 # in bytes, should always be equal to in1_size
trace_size = 8192

data_size = in1_size
aie_py_src=${targetname}.py
use_alt?=0

ifeq (${use_alt}, 1)
aie_py_src=${targetname}_alt.py
endif

.PHONY: all template clean

all: build/final_${data_size}.xclbin

build/aie2_lineBased_8b_${data_size}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${device} ${in1_size} ${out_size} 0 > $@

build/aie2_trace_lineBased_8b_${data_size}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${device} ${in1_size} ${out_size} ${trace_size} > $@

build/%.cc.o: %.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else
	echo "Device type not supported"
endif
	
build/final_${data_size}.xclbin: build/aie2_lineBased_8b_${data_size}.mlir build/passThrough.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts_${data_size}.txt $(<:%=../%)

build/final_trace_${data_size}.xclbin: build/aie2_trace_lineBased_8b_${data_size}.mlir build/passThrough.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts_${data_size}.txt $(<:%=../%)


${targetname}_${data_size}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname}  -DIN1_SIZE=${in1_size} -DOUT_SIZE=${out_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}_${data_size}.exe build/final_${data_size}.xclbin build/insts_${data_size}.txt
	${powershell} ./$< -x build/final_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE 

run_py: build/final_${data_size}.xclbin build/insts_${data_size}.txt
	${powershell} python3 ${srcdir}/test.py -x build/final_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE -i1s ${in1_size} -os ${out_size}

trace: ${targetname}_${data_size}.exe build/final_trace_${data_size}.xclbin build/insts_${data_size}.txt 
	${powershell} ./$< -x build/final_trace_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../utils/parse_trace.py --filename trace.txt --mlir build/aie2_trace_lineBased_8b_${data_size}.mlir --colshift 1 > trace_passthrough_kernel.json
	${srcdir}/../../utils/get_trace_summary.py --filename trace_passthrough_kernel.json

trace_py: build/final_trace_${data_size}.xclbin build/insts_${data_size}.txt
	${powershell} python3 ${srcdir}/test.py -x build/final_trace_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE -t ${trace_size} -i1s ${in1_size} -os ${out_size}
	${srcdir}/../../utils/parse_trace.py --filename trace.txt --mlir build/aie2_trace_lineBased_8b_${data_size}.mlir --colshift 1 > trace_passthrough_kernel.json
	${srcdir}/../../utils/get_trace_summary.py --filename trace_passthrough_kernel.json

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean:
	rm -rf build _build ${targetname}*.exe
