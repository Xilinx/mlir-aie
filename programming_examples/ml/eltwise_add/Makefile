##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../makefile-common

all: build/final.xclbin

targetname = myEltwiseAdd

build/add.o:
	mkdir -p ${@D}
	cd ${@D} &&	xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -I. -c ../../../../aie_kernels/aie2/add.cc -o ${@F}

build/aie.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< > $@

build/final.xclbin: build/aie.mlir build/add.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt ${<F}

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

run_py: build/final.xclbin build/insts.txt
	${powershell} python3 test.py -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

clean:
	rm -rf build _build ${targetname}.exe
