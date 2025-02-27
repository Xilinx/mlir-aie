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

targetname = softmax
trace_size = 8192

aie_py_src=${targetname}.py
use_alt?=0

ifeq (${use_alt}, 1)
aie_py_src=${targetname}_alt.py
endif

all: build/final.xclbin build/insts.txt

build/dut.cc: ${srcdir}/bf16_softmax.mlir
	mkdir -p ${@D}
	cd ${@D} &&	aie-opt $< -affine-super-vectorize="virtual-vector-size=16 test-fastest-varying=0 vectorize-reductions=true" --convert-vector-to-aievec="aie-target=aie2" -lower-affine | aie-translate -aie2=true --aievec-to-cpp -o ${@F}

build/dut.o: build/dut.cc
	cd ${@D} &&	${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I../../../../aie_runtime_lib/AIE2 -c ${<F} -o ${@F}

build/lut_based_ops.o: ../../../aie_runtime_lib/AIE2/lut_based_ops.cpp
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I. -c $(<:%=../%) -o ${@F}

build/softmax.o: bf16_softmax.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I. -I../../../../aie_runtime_lib/AIE2 -c $< -o ${@F}

build/kernels.a: build/softmax.o 
	ar rvs $@ $+

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< > $@

build/aie_trace.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${trace_size} > $@


build/final.xclbin: build/aie.mlir build/kernels.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
    	--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				--aie-generate-npu --npu-insts-name=insts.txt $(<:%=../%)

build/final_trace.xclbin: build/aie_trace.mlir build/kernels.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
    	--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				--aie-generate-npu --npu-insts-name=insts.txt $(<:%=../%)


${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

profile: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE -p results.csv

trace: ${targetname}.exe build/final_trace.xclbin
	${powershell} ./$< -x build/final_trace.xclbin -i build/insts.txt -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../utils/parse_trace.py --filename trace.txt --mlir build/aie_trace.mlir --colshift 1 > trace_softmax.json
	${srcdir}/../../utils/get_trace_summary.py --filename trace_softmax.json

clean_trace:
	rm -rf trace.txt trace_softmax.json

clean: clean_trace
	rm -rf build _build ${targetname}.exe 

