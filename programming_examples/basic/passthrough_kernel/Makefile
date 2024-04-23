##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../makefile-common

VPATH := ../../../aie_kernels/generic

PASSTHROUGH_SIZE = 4096

targetname = passThroughKernel

.PHONY: all template clean

all: build/final_${PASSTHROUGH_SIZE}.xclbin

build/aie2_lineBased_8b_${PASSTHROUGH_SIZE}.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< ${PASSTHROUGH_SIZE} > $@

build/passThrough.cc.o: passThrough.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBIT_WIDTH=8 -c $(<:%=../%) -o ${@F}
	
build/final_${PASSTHROUGH_SIZE}.xclbin: build/aie2_lineBased_8b_${PASSTHROUGH_SIZE}.mlir build/passThrough.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--xclbin-name=${@F} --npu-insts-name=insts.txt $(<:%=../%)

${targetname}.exe: test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake .. -DTARGET_NAME=${targetname} -DPASSTHROUGH_SIZE=${PASSTHROUGH_SIZE}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final_${PASSTHROUGH_SIZE}.xclbin build/insts.txt
	${powershell} ./$< -x build/final_${PASSTHROUGH_SIZE}.xclbin -i build/insts.txt -k MLIR_AIE 

run-g: ${targetname}.exe build/final_${PASSTHROUGH_SIZE}.xclbin build/insts.txt
	${powershell} ./$< -x build/final_${PASSTHROUGH_SIZE}.xclbin -i build/insts.txt -k MLIR_AIE -t 8192

run_py: build/final_${PASSTHROUGH_SIZE}.xclbin build/insts.txt
	${powershell} python3 test.py -s ${PASSTHROUGH_SIZE} -x build/final_${PASSTHROUGH_SIZE}.xclbin -i build/insts.txt -k MLIR_AIE

trace:
	../../utils/parse_eventIR.py --filename trace.txt --mlir build/aie.mlir --colshift 1 > parse_eventIR_vs.json

clean_trace:
	rm -rf tmpTrace trace.txt

clean:
	rm -rf build _build ${targetname}.exe
