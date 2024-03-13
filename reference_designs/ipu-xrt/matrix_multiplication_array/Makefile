##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../makefile-common

M?=512
K?=512
N?=512
runargs?=-v 1 --verify 0 --warmup 10 --iters 10

xclbin_sign=/opt/xilinx/xrt/amdxdna/setup_xclbin_firmware.sh 
targetname = matrixMultiplication

mlir_target=build/aie_${M}x${K}x${N}.mlir
xclbin_target=build/final_${M}x${K}x${N}.xclbin
insts_target=build/insts_${M}x${K}x${N}.txt

.PHONY: all
all: ${xclbin_target} ${insts_target} ${targetname}.exe sign

build/%.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBIT_WIDTH=8 -c $(<:%=../%) -o ${@F}

${mlir_target}: aie2.py
	mkdir -p ${@D}
	python3 $< -M $M -K $K -N $N > $@

${xclbin_target}: ${mlir_target} build/mm.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-ipu --ipu-insts-name=${insts_target:build/%=%} $(<:%=../%)

${targetname}.exe: test.cpp ../matrix_multiplication.h
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake -E env CXXFLAGS="-std=c++23 -ggdb" cmake .. -D CMAKE_C_COMPILER=gcc-13 -D CMAKE_CXX_COMPILER=g++-13 -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

.PHONY: sign
sign: ${xclbin_target}
	${xclbin_sign} -dev Phoenix -xclbin $<

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${insts_target} #sign
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -M $M -K $K -N $N ${runargs}

.PHONY: clean
clean:
	rm -rf build _build ${targetname}.exe
