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
VPATH := ${srcdir}/../../../aie_kernels/aie2

include $(srcdir)/../../makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = dequant
in1_size = 65536
out_size = 131072
trace_size = 65536
group_size ?= 32
CHESS ?= false

data_size = $(in1_size)
aie_py_src=${targetname}.py

.PHONY: all template clean

all: build/final.xclbin

build/expand.cc.o: expand.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -DGROUP_SIZE=$(group_size) -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -DGROUP_SIZE=$(group_size) -c $< -o ${@F}
else
	echo "Device type not supported"
endif

build/aie.mlir: $(srcdir)/$(aie_py_src)
	mkdir -p $(@D)
	python3 $< -d $(devicename) -i1s $(in1_size) -os $(out_size) -gs $(group_size) > $@

build/aie_trace.mlir: $(srcdir)/$(aie_py_src)
	mkdir -p $(@D)
	python3 $< -d $(devicename) -i1s $(in1_size) -os $(out_size) -t $(trace_size) -gs $(group_size) > $@

build/final.xclbin: build/aie.mlir build/expand.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

build/final_trace.xclbin: build/aie_trace.mlir build/expand.cc.o
	mkdir -p .
	cd build && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)


${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname}  -DIN1_SIZE=${in1_size} -DOUT_SIZE=${out_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin build/insts.bin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE 

run_py: build/final.xclbin build/insts.bin
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -i1s ${in1_size} -os ${out_size}

trace: ${targetname}.exe build/final_trace.xclbin build/insts.bin
	${powershell} ./$< -x build/final_trace.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../utils/parse_trace.py --input trace.txt --mlir build/aie_trace.mlir --output trace_dequant.json
	${srcdir}/../../utils/get_trace_summary.py --input trace_dequant.json

trace_py: build/final_trace.xclbin build/insts.bin
	${powershell} python3 ${srcdir}/test.py -x build/final_trace.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size} -i1s ${in1_size} -os ${out_size}
	${srcdir}/../../utils/parse_trace.py --input trace.txt --mlir build/aie_trace.mlir --output trace_${targetname}.json
	${srcdir}/../../utils/get_trace_summary.py --input trace_${targetname}.json

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean: clean_trace
	rm -rf build _build ${targetname}*.exe
