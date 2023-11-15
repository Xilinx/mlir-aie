##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../makefile-common

VPATH := ../vision_kernels

#all: build/final_1080.xclbin
#all: build/final_vga.xclbin
all: build/final_tiny.xclbin

build/%.cc.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc ${CHESSCC2_FLAGS} -DBIT_WIDTH=8 -c $(<:%=../%) -o ${@F}

build/combined_gray2rgba_addWeighted.a: build/gray2rgba.cc.o build/addWeighted.cc.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^)

build/aie2_lineBased_8b_%.mlir: aie2_lineBased_8b_%.py
	mkdir -p ${@D}
	python3 $< > build/tmp.mlir
	aie-opt --canonicalize build/tmp.mlir -o $@
	rm build/tmp.mlir

build/final_%.xclbin: build/aie2_lineBased_8b_%.mlir build/rgba2gray.cc.o build/gray2rgba.cc.o build/filter2d.cc.o build/threshold.cc.o build/addWeighted.cc.o build/combined_gray2rgba_addWeighted.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-cdo --aie-generate-ipu --no-compile-host --xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

# build/test.exe: test.cpp
# 	mkdir -p ${@D}
# 	$(CXX) $(LD_FLAGS) $(OpenCV_LIBS) ../utils/OpenCVUtils.cpp ../utils/xrtUtils.cpp $< -o $@ $(CFLAGS) $(INC) $(LD_PATHS) 

clean:
	rm -rf build
