#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common

VPATH := ${srcdir}/../../../../aie_kernels/aie2

all: build/final_bn_1.xclbin

build/aie2_%.mlir: ${srcdir}/aie2_%.py ${srcdir}/aie2_bottleneckA.py ${srcdir}/aie2_bottleneck0.py ${srcdir}/aie2_bottleneck0And1.py ${srcdir}/aie2_bottleneck2And3.py
	mkdir -p ${@D}
	python3 $< > $@

# insts_.txt: build/${mlirFileName}.mlir
# 	aiecc.py -v --aie-only-generate-npu --npu-insts-name=$@ $<
# ****************************************************************************************
# bn2

build/bn2_conv2dk1_fused_relu.o: bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DINT8_ACT -DBN2 -c $< -o ${@F}
#	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DINT8_ACT -c $< -o $@

# bn3
build/bn3_conv2dk1_fused_relu.o: bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DINT8_ACT -DBN3 -c $< -o ${@F}
#	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DINT8_ACT -c $< -o $@

build/combined_bn2_bn3.a: build/conv2dk3_dw_stride1.o build/bn2_conv2dk1_fused_relu.o  build/conv2dk1_skip.o build/bn3_conv2dk1_fused_relu.o
	mkdir -p ${@D}
	ar rvs $@ $^ $(word 2,$^) $(word 3,$^) $(word 4,$^) $(word 5,$^)
# ****************************************************************************************

build/conv2dk1_i8.o: bn_conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS}  -DSCALAR  -DREGULAR -c $< -o ${@F}

build/conv2dk1_fused_relu.o: bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS}  -DINT8_ACT -DREGULAR -c $< -o ${@F}
#	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DINT8_ACT -c $< -o $@


build/conv2dk3_dw_stride2.o: bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DREGULAR  -DSCALAR -DSTRIDE2 -c $< -o ${@F}
#	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DUINT8_ACT -DSTRIDE2 -c $< -o $@

build/conv2dk3_dw_stride1.o: bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DREGULAR  -DSCALAR  -DSTRIDE1 -c $< -o ${@F}

build/conv2dk1_skip.o: bn_conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DREGULAR -DSCALAR -c $< -o ${@F}

build/conv2dk1_skipui8.o: bn_conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DREGULAR -DSCALAR -DUNSIGNED_SKIP -c $< -o ${@F}

build/combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a: build/conv2dk1_fused_relu.o build/conv2dk3_dw_stride2.o build/conv2dk1_i8.o  
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

build/combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a: build/conv2dk1_fused_relu.o build/conv2dk3_dw_stride1.o build/conv2dk1_i8.o  
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

build/combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a:  build/conv2dk1_fused_relu.o build/conv2dk3_dw_stride1.o build/conv2dk1_skip.o
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

build/combined_conv2dk3dwstride1_conv2dk1skip.a: build/conv2dk3_dw_stride1.o build/conv2dk1_skipui8.o 
	mkdir -p ${@D}
	ar rvs $@ $^

build/combined_bn0_bn1.a: build/conv2dk3_dw_stride1.o build/conv2dk1_skipui8.o build/conv2dk1_fused_relu.o build/conv2dk3_dw_stride2.o build/conv2dk1_i8.o 
	mkdir -p ${@D}
	ar rvs $@ $^ $(word 2,$^) $(word 3,$^) $(word 4,$^) $(word 5,$^)


build/final_%.xclbin build/inst_%.txt &: build/aie2_%.mlir build/combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a build/combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a build/combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a build/combined_conv2dk3dwstride1_conv2dk1skip.a build/combined_bn0_bn1.a  build/combined_bn2_bn3.a
	cd build && aiecc.py -v --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--basic-alloc-scheme \
		--xclbin-name=${@F} --npu-insts-name=insts_$*.txt ${<F}

clean:
	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj log .xclbin sim \
		chess* *.o insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe

run_py_0: build/final_bn_0.xclbin build/insts_bn_0.txt build/aie2_bn_0.mlir
	${powershell} python3 ${srcdir}/test_bn_0.py -x build/final_bn_0.xclbin -i build/insts_bn_0.txt -k MLIR_AIE

run_py_1: build/final_bn_1.xclbin build/insts_bn_1.txt build/aie2_bn_1.mlir
	${powershell} python3 ${srcdir}/test_bn_1.py -x build/final_bn_1.xclbin -i build/insts_bn_1.txt -k MLIR_AIE


run_py_2: build/final_bn_2.xclbin build/insts_bn_2.txt build/aie2_bn_2.mlir
	${powershell} python3 ${srcdir}/test_bn_2.py -x build/final_bn_2.xclbin -i build/insts_bn_2.txt -k MLIR_AIE

run_py_3: build/final_bn_3.xclbin build/insts_bn_3.txt build/aie2_bn_3.mlir
	${powershell} python3 ${srcdir}/test_bn_3.py -x build/final_bn_3.xclbin -i build/insts_bn_3.txt -k MLIR_AIE

run_py_4: build/final_bn_4.xclbin build/insts_bn_4.txt build/aie2_bn_4.mlir
	${powershell} python3 ${srcdir}/test_bn_4.py -x build/final_bn_4.xclbin -i build/insts_bn_4.txt -k MLIR_AIE

run_py_5: build/final_bn_5.xclbin build/insts_bn_5.txt build/aie2_bn_5.mlir
	${powershell} python3 ${srcdir}/test_bn_5.py -x build/final_bn_5.xclbin -i build/insts_bn_5.txt -k MLIR_AIE

run_py_6: build/final_bn_6.xclbin build/insts_bn_6.txt build/aie2_bn_6.mlir
	${powershell} python3 ${srcdir}/test_bn_6.py -x build/final_bn_6.xclbin -i build/insts_bn_6.txt -k MLIR_AIE


run_py_7: build/final_bn_7.xclbin build/insts_bn_7.txt build/aie2_bn_7.mlir
	${powershell} python3 ${srcdir}/test_bn_7.py -x build/final_bn_7.xclbin -i build/insts_bn_7.txt -k MLIR_AIE

run_py_8: build/final_bn_8.xclbin build/insts_bn_8.txt build/aie2_bn_8.mlir
	${powershell} python3 ${srcdir}/test_bn_8.py -x build/final_bn_8.xclbin -i build/insts_bn_8.txt -k MLIR_AIE

run_py_9: build/final_bn_9.xclbin build/insts_bn_9.txt build/aie2_bn_9.mlir
	${powershell} python3 ${srcdir}/test_bn_9.py -x build/final_bn_9.xclbin -i build/insts_bn_9.txt -k MLIR_AIE

run_py_0_1: build/final_bn_0_1.xclbin build/insts_bn_0_1.txt build/aie2_bn_0_1.mlir
	${powershell} python3 ${srcdir}/test_bn_0_1.py -x build/final_bn_0_1.xclbin -i build/insts_bn_0_1.txt -k MLIR_AIE


run_py_2_3: build/final_bn_2_3.xclbin build/insts_bn_2_3.txt build/aie2_bn_2_3.mlir
	${powershell} python3 ${srcdir}/test_bn_2_3.py -x build/final_bn_2_3.xclbin -i build/insts_bn_2_3.txt -k MLIR_AIE