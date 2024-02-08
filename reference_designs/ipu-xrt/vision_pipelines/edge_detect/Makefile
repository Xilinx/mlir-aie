##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../makefile-common

VPATH := ../vision_kernels

EDGEDETECT_WIDTH = 1920
EDGEDETECT_HEIGHT = 1080

#EDGEDETECT_WIDTH = 640
#EDGEDETECT_HEIGHT = 480

#EDGEDETECT_WIDTH = 64
#EDGEDETECT_HEIGHT = 36

targetname = edgeDetect

default: peano

all: peano xchesscc peanoxbridge

mlir: build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir
peano: build/final_${EDGEDETECT_WIDTH}.xclbin
xchesscc: build_xchesscc/final_${EDGEDETECT_WIDTH}.xclbin
peanoxbridge: build_peanoxbridge/final_${EDGEDETECT_WIDTH}.xclbin

PEANO = /root/acdc/build/install/peano/bin/clang++
build/%.cc.o: %.cc
	@mkdir -p ${@D}
	cd ${@D} && ${PEANO} ${CLANG_FLAGS} -DBIT_WIDTH=8 -c $(<:%=../%) -o ${@F}

build/combined_gray2rgba_addWeighted.a: build/gray2rgba.cc.o build/addWeighted.cc.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^)

build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir: aie2_edgeDetect.py
	mkdir -p ${@D}
	python3 $< ${EDGEDETECT_WIDTH} ${EDGEDETECT_HEIGHT} > $@

build/final_${EDGEDETECT_WIDTH}.xclbin: build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir build/rgba2gray.cc.o build/gray2rgba.cc.o build/filter2d.cc.o build/threshold.cc.o build/addWeighted.cc.o build/combined_gray2rgba_addWeighted.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

XCHESSCC_OBJS = build_xchesscc/rgba2gray.cc.o \
				build_xchesscc/gray2rgba.cc.o \
				build_xchesscc/filter2d.cc.o \
				build_xchesscc/threshold.cc.o \
				build_xchesscc/addWeighted.cc.o \
				build_xchesscc/combined_gray2rgba_addWeighted.a

build_xchesscc/%.cc.o: %.cc
	@mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} +w ${@F}.work -DBIT_WIDTH=8 -c $(<:%=../%) -o ${@F}

build_xchesscc/combined_gray2rgba_addWeighted.a: build_xchesscc/gray2rgba.cc.o build_xchesscc/addWeighted.cc.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^)

build_xchesscc/final_${EDGEDETECT_WIDTH}.xclbin: build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir ${XCHESSCC_OBJS}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --xchesscc --xbridge --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

PEANOXBRIDGE_OBJS = build_peanoxbridge/rgba2gray.cc.o \
				build_peanoxbridge/gray2rgba.cc.o \
				build_peanoxbridge/filter2d.cc.o \
				build_peanoxbridge/threshold.cc.o \
				build_peanoxbridge/addWeighted.cc.o \
				build_peanoxbridge/combined_gray2rgba_addWeighted.a
build_peanoxbridge/%.cc.o: %.cc
	@mkdir -p ${@D}
	cd ${@D} && ${PEANO} ${CLANG_CHESSLIBC_FLAGS} -DBIT_WIDTH=8 -c $(<:%=../%) -o ${@F}

build_peanoxbridge/combined_gray2rgba_addWeighted.a: build_peanoxbridge/gray2rgba.cc.o build_peanoxbridge/addWeighted.cc.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^)

build_peanoxbridge/final_${EDGEDETECT_WIDTH}.xclbin: build/aie2_lineBased_8b_${EDGEDETECT_WIDTH}.mlir ${PEANOXBRIDGE_OBJS}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --xbridge --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

${targetname}.exe: test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake .. -DTARGET_NAME=${targetname} -DEDGEDETECT_WIDTH=${EDGEDETECT_WIDTH} -DEDGEDETECT_HEIGHT=${EDGEDETECT_HEIGHT} 
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final_${EDGEDETECT_WIDTH}.xclbin build/insts.txt
	${powershell} ./$< -x build/final_${EDGEDETECT_WIDTH}.xclbin -i build/insts.txt -k MLIR_AIE

clean:
	rm -rf build _build ${targetname}.exe
