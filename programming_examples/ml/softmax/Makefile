##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

aie2_runtime_dir = ${MLIR_AIE_DIR}/aie_runtime_lib/AIE2

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

ifeq ($(devicename),npu2)
VPATH :=${srcdir}/../../../aie_kernels/aie2p
else
VPATH :=${srcdir}/../../../aie_kernels/aie2
endif

targetname = softmax
trace_size = 8192

aie_py_src=${targetname}.py
use_placed?=0
use_whole_array?=0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

ifeq (${use_whole_array}, 1)
aie_py_src=${targetname}_whole_array_placed.py
endif

all: build/final.xclbin build/insts.bin

ifeq ($(devicename),npu)
build/lut_based_ops.o: ${aie2_runtime_dir}/lut_based_ops.cpp
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I. -c $< -o ${@F}
endif


build/softmax.o: ${VPATH}/softmax.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I. -I${aie2_runtime_dir} -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	echo "Device type not supported"
endif

ifeq ($(devicename),npu)
build/kernels.a: build/softmax.o build/lut_based_ops.o
	ar rvs $@ $+
else ifeq ($(devicename),npu2)
build/kernels.a: build/softmax.o
	ar rvs $@ $+
else
	echo "Device type not supported"
endif

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/aie_trace.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} ${trace_size} > $@


build/final.xclbin: build/aie.mlir build/kernels.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    	--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

build/final_trace.xclbin: build/aie_trace.mlir build/kernels.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    	--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

ifeq ($(devicename),npu)
run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE --npu 1
else ifeq ($(devicename),npu2)
run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE --npu 2
endif

profile: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -p results.csv

trace: ${targetname}.exe build/final_trace.xclbin
	${powershell} ./$< -x build/final_trace.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../utils/parse_trace.py --input trace.txt --mlir build/aie_trace.mlir --output trace_softmax.json
	${srcdir}/../../utils/get_trace_summary.py --input trace_softmax.json

clean:
	rm -rf build _build ${targetname}.exe

