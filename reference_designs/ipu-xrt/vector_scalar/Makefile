##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../makefile-common

all: build/final.xclbin build/insts.txt

#run: all test.exe
#	pushd build; ./test.exe -x ./final.xclbin -k MLIR_AIE -i ./insts.txt; popd

build/%.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBIT_WIDTH=8 -c $(<:%=../%) -o ${@F}

build/aie.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< > build/tmp.mlir
	aie-opt --canonicalize build/tmp.mlir -o $@
	rm build/tmp.mlir

build/final.xclbin: build/aie.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-ipu --ipu-insts-name=insts.txt $(<:%=../%)

# build/test.exe: test.cpp
# 	mkdir -p ${@D}
# 	$(CXX) $(LD_FLAGS) $< -o $@ $(CFLAGS) $(INC) $(LD_PATHS) 

vectorScalar.exe: test.cpp build/final.xclbin
	rm -rf _build
	mkdir _build
	cd _build && ${powershell} cmake .. 
	cd _build && ${powershell} cmake --build . --config Release
	cp _build/Release/$@ ./

run: vectorScalar.exe
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

clean:
	rm -rf build _build vectorScalar.exe
