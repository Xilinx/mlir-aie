##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../makefile-common

all: hello_world_elfstrings.csv

targetname = helloWorld

build/%.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $(<:%=../%) -o ${@F}

build/hello_world.mlir: hello_world.py
	mkdir -p ${@D}
	python3 $< > $@

build/hello_world.xclbin: build/hello_world.mlir build/kernel.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

hello_world_elfstrings.csv: build/hello_world.xclbin
	python3 elfStringParser.py --input ./build --output $@

${targetname}.exe: test.cpp
	mkdir -p ${@D}
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake .. -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe hello_world_elfstrings.csv
	${powershell} ./$< -x build/hello_world.xclbin -i build/insts.txt \
		-k MLIR_AIE -e $(word 2,$^)

clean:
	rm -rf build _build *.csv ${powershell}.exe
