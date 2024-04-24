##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

targetname = passThroughKernel
VPATH := ${srcdir}/../../../aie_kernels/generic
data_size = 4096
trace_size = 8192
PASSTHROUGH_SIZE = data_size

.PHONY: all template clean

all: build/final_${data_size}.xclbin

build/aie2_lineBased_8b_${data_size}.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${data_size} ${trace_size} > $@

build/passThrough.cc.o: passThrough.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
	
build/final_${data_size}.xclbin: build/aie2_lineBased_8b_${data_size}.mlir build/passThrough.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--xclbin-name=${@F} --npu-insts-name=insts.txt $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname} -DPASSTHROUGH_SIZE=${data_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final_${data_size}.xclbin build/insts.txt
	${powershell} ./$< -x build/final_${data_size}.xclbin -i build/insts.txt -k MLIR_AIE 

run-g: ${targetname}.exe build/final_${data_size}.xclbin build/insts.txt
	${powershell} ./$< -x build/final_${data_size}.xclbin -i build/insts.txt -k MLIR_AIE -t ${trace_size}

run_py: build/final_${data_size}.xclbin build/insts.txt
	${powershell} python3 ${srcdir}/test.py -s ${data_size} -x build/final_${data_size}.xclbin -i build/insts.txt -k MLIR_AIE

trace:
	../../utils/parse_eventIR.py --filename trace.txt --mlir build/aie2_lineBased_8b_${data_size}.mlir --colshift 1 > parse_eventIR_vs.json

clean_trace:
	rm -rf tmpTrace trace.txt parse_eventIR_vs.json

clean:
	rm -rf build _build ${targetname}.exe
