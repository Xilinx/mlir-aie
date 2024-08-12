#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common

VPATH := ${srcdir}/../../../../aie_kernels/aie2

OBJ = build/combined_bn2_bn3.a 


all: clean build/final_bn_2_3.xclbin run_py_bn_2_3

build/aie2_bn_2_3.mlir: ${srcdir}/aie2_bn_2_3.py 
	mkdir -p ${@D}
	python3 $< > $@

# insts_bn_2_3.txt: build/aie2_bn_2_3.mlir
# 	aiecc.py -v --aie-only-generate-npu --npu-insts-name=$@ $<

# ****************************************************************************************
# bn2

build/bn2_conv2dk1_fused_relu.o: bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DINT8_ACT -DBN2 -c $< -o ${@F}
#	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DINT8_ACT -c $< -o $@

build/conv2dk3_dw_stride1.o: bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DREGULAR  -DSCALAR  -DSTRIDE1 -c $< -o ${@F}


build/conv2dk1_skip.o: bn_conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DREGULAR -DSCALAR -c $< -o ${@F}

build/bn3_conv2dk1_fused_relu.o: bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DINT8_ACT -DBN3 -c $< -o ${@F}
#	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DINT8_ACT -c $< -o $@

build/conv2dk3_dw_stride2.o: bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DREGULAR  -DSCALAR -DSTRIDE2 -c $< -o ${@F}
#	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DUINT8_ACT -DSTRIDE2 -c $< -o $@

build/conv2dk1_i8.o: bn_conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS}  -DSCALAR  -DREGULAR -c $< -o ${@F}

build/combined_bn2_bn3.a: build/conv2dk3_dw_stride1.o build/bn2_conv2dk1_fused_relu.o  build/conv2dk1_skip.o build/bn3_conv2dk1_fused_relu.o build/conv2dk3_dw_stride2.o
	mkdir -p ${@D}
	ar rvs $@ $^ $(word 2,$^) $(word 3,$^) $(word 4,$^) $(word 5,$^)
# ****************************************************************************************

build/final_bn_2_3.xclbin: build/aie2_bn_2_3.mlir  $(OBJ)
	cd build && aiecc.py -v --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--basic-alloc-scheme \
		--xclbin-name=${@F} --npu-insts-name=insts.txt ${<F}

run_py_bn_2_3: clean build/final_bn_2_3.xclbin build/aie2_bn_2_3.mlir
	${powershell} python3 ${srcdir}/test_bn_2_3.py -x build/final_bn_2_3.xclbin -i build/insts.txt -k MLIR_AIE

# clean:
# 	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj log .xclbin sim \
# 		chess* *.o insts.txt \
# 		*.log aie_partition.json *.bin BOOT.BIN _x test.exe
clean:
	rm -rf build/*.elf* build/*.lst build/*.bif build/*.mlir.prj build/*.mlir log build/.xclbin build/sim \
		build/insts.txt \
		build/*.log aie_partition.json build/*.bin build/BOOT.BIN _x build/test.exe