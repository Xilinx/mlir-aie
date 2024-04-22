##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../makefile-common

targetname = testExp

all: build/final.xclbin build/insts.txt

VPATH := ../../../aie_kernels/aie2

build/exp.o: bf16_exp.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -I../../../../aie_runtime_lib/AIE2 -c $(<:%=../%) -o ${@F}

build/lut_based_ops.o: ../../../aie_runtime_lib/AIE2/lut_based_ops.cpp
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -I. -c $(<:%=../%) -o ${@F}

build/kernels.a: build/exp.o build/lut_based_ops.o
	ar rvs $@ $+

build/aie.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< > $@

build/final.xclbin: build/aie.mlir build/kernels.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-ipu --ipu-insts-name=insts.txt $(<:%=../%)

${targetname}.exe: test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake .. -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin build/insts.txt 
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

clean: 
	rm -rf build _build ${targetname}.exe 

