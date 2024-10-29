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

targetname = vectorScalar
data_size = 4096
trace_size = 8192
CHESS ?= true

all: build/final_${data_size}.xclbin build/insts_${data_size}.txt

kristof: build/insts_${data_size}.txt

build/%.o: %.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}; 
else 
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}; 
endif

build/aie_${data_size}.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${data_size} 0 > $@

build/aie_trace_${data_size}.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< ${data_size} ${trace_size} > $@

#build/insts_${data_size}.txt: build/final_${data_size}.xclbin
build/final_${data_size}.xclbin: build/aie_${data_size}.mlir build/scale.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=insts_${data_size}.txt $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
    	  --no-xchesscc --no-xbridge \
				--aie-generate-npu --npu-insts-name=insts_${data_size}.txt $(<:%=../%)
endif

build/final_trace_${data_size}.xclbin: build/aie_trace_${data_size}.mlir build/scale.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=insts_${data_size}.txt $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
				--aie-generate-npu --npu-insts-name=insts_${data_size}.txt $(<:%=../%)
endif

${targetname}_${data_size}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname}_${data_size} -DVECTORSCALARMUL_SIZE=${data_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}_${data_size}.exe $@
else
	cp _build/${targetname}_${data_size} $@ 
endif

run: ${targetname}_${data_size}.exe build/final_${data_size}.xclbin build/insts_${data_size}.txt 
	${powershell} ./$< -x build/final_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE

run_py: build/final_${data_size}.xclbin build/insts_${data_size}.txt
	${powershell} python3 ${srcdir}/test.py -x build/final_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE -s ${data_size}

trace: ${targetname}_${data_size}.exe build/final_trace_${data_size}.xclbin build/insts_${data_size}.txt 
	${powershell} ./$< -x build/final_trace_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE -t ${trace_size}
	../../utils/parse_trace.py --filename trace.txt --mlir build/aie_trace_${data_size}.mlir --colshift 1 > trace_vs.json

trace_py: build/final_trace_${data_size}.xclbin build/insts_${data_size}.txt
	${powershell} python3 ${srcdir}/test.py -x build/final_trace_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE -t ${trace_size} -s ${data_size}
	../../utils/parse_trace.py --filename trace.txt --mlir build/aie_trace_${data_size}.mlir --colshift 1 > trace_vs.json


clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean: clean_trace
	rm -rf build _build ${targetname}*.exe
