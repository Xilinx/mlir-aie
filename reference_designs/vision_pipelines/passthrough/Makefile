##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../../test/ipu-xrt/makefile-common

VPATH := ../vision_kernels

.PHONY: all template clean

all: build/final_tiny.xclbin

all_1080: build/final_1080.xclbin

all_8k: build/final_8k.xclbin

build/passThrough.cc.o: passThrough.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc -d ${CHESSCC2_FLAGS} -DBIT_WIDTH=8 -c $(<:%=../%) -o ${@F}
	
build/final_%.xclbin: aie2_lineBased_8b_%.mlir build/passThrough.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-cdo --aie-generate-ipu --no-compile-host --xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

clean:
	rm -rf build

#build/test.exe: test.cpp
#	mkdir -p ${@D}
#	cd ${@D} && $(CXX) $(LD_FLAGS) $(OpenCV_LIBS) ../../utils/OpenCVUtils.cpp ../../utils/xrtUtils.cpp $(<:%=../%) -o ${@F} $(CFLAGS) $(INC) $(LD_PATHS) 