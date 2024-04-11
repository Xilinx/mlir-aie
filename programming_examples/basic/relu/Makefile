##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../makefile-common

targetname = testRelu

all: build/final.xclbin build/insts.txt

build/bf16_relu.o: ${REPO_ROOT}/aie_kernels/relu.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -I. -I${REPO_ROOT}/my_install/mlir_aie/aie_runtime_lib/AIE2 -c $< -o ${@F}

build/aie.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< > $@

build/final.xclbin: build/aie.mlir build/bf16_relu.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-ipu --ipu-insts-name=insts.txt $(<:%=../%)

${targetname}.exe: test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake -E env CXXFLAGS="-std=c++23" cmake .. -D CMAKE_C_COMPILER=gcc-13 -D CMAKE_CXX_COMPILER=g++-13 -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin build/insts.txt 
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

run_g: ${targetname}.exe build/final.xclbin build/insts.txt 
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE --trace_sz 65536


trace:
	../../utils/parse_eventIR.py --filename trace.txt --mlir build/aie.mlir --colshift 1 > parse_eventIR_vs.json

clean_trace:
	rm -rf tmpTrace trace.txt

clean: clean_trace
	rm -rf build _build ${targetname}.exe

