#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common

VPATH := ${srcdir}/../../../../aie_kernels/aie2

CHESS ?= true

#all: build/final_bn_0.xclbin
all: build/final_bn_chain.xclbin

#build/aie2_%.mlir: ${srcdir}/aie2_%.py ${srcdir}/aie2_bottleneckA.py ${srcdir}/aie2_bottleneck0.py ${srcdir}/aie2_bottleneck0And1.py ${srcdir}/aie2_bottleneck2And3.py
#build/aie2_%.mlir: ${srcdir}/aie2_%.py ${srcdir}/aie2_bottleneckA.py ${srcdir}/aie2_bottleneck0Static.py ${srcdir}/aie2_bottleneck0And1.py ${srcdir}/aie2_bottleneck2And3.py
build/aie2_%.mlir: ${srcdir}/aie2_%.py ${srcdir}/aie2_bottleneckAStatic.py ${srcdir}/aie2_bottleneck0Static.py ${srcdir}/aie2_bottleneck8And9Static.py
	mkdir -p ${@D}
	python3 $< > $@

# insts_.bin: build/${mlirFileName}.mlir
# 	aiecc.py -v --aie-only-generate-npu --npu-insts-name=$@ $<

# ****************************************************************************************
# # bn2
# build/bn2_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
# 	mkdir -p ${@D}
# ifeq ($(CHESS), true)
# 	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DINT8_ACT -DBN2 -c $< -o ${@F}
# else
# 	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DINT8_ACT -DBN2 -c $< -o ${@F}
# endif

# # bn3
# build/bn3_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
# 	mkdir -p ${@D}
# ifeq ($(CHESS), true)
# 	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DINT8_ACT -DBN3 -c $< -o ${@F}
# else
# 	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DINT8_ACT -DBN3 -c $< -o ${@F}
# endif

# build/combined_bn2_bn3.a: build/conv2dk3_dw_stride1.o build/bn2_conv2dk1_fused_relu.o  build/conv2dk1_skip.o build/bn3_conv2dk1_fused_relu.o
# 	mkdir -p ${@D}
# 	ar rvs $@ $^ $(word 2,$^) $(word 3,$^) $(word 4,$^) $(word 5,$^)

# ****************************************************************************************

build/conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS}  -DSCALAR  -DREGULAR -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DSCALAR -DREGULAR -c $< -o ${@F}
endif

build/conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS}  -DINT8_ACT -DREGULAR -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DINT8_ACT -DREGULAR -c $< -o ${@F}
endif


build/conv2dk3_dw_stride2.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DREGULAR  -DSCALAR -DSTRIDE2 -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DREGULAR  -DSCALAR -DSTRIDE2 -c $< -o ${@F}
endif

build/conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DREGULAR  -DSCALAR  -DSTRIDE1 -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DREGULAR  -DSCALAR  -DSTRIDE1 -c $< -o ${@F}
endif

build/conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DREGULAR -DSCALAR -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DREGULAR -DSCALAR -c $< -o ${@F}
endif

build/conv2dk1_skipui8.o: bottleneck/bn_conv2dk1_skip.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DREGULAR -DSCALAR -DUNSIGNED_SKIP -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DREGULAR -DSCALAR -DUNSIGNED_SKIP -c $< -o ${@F}
endif

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

# build/combined_bn0_bn1.a: build/conv2dk3_dw_stride1.o build/conv2dk1_skipui8.o build/conv2dk1_fused_relu.o build/conv2dk3_dw_stride2.o build/conv2dk1_i8.o 
# 	mkdir -p ${@D}
# 	ar rvs $@ $^ $(word 2,$^) $(word 3,$^) $(word 4,$^) $(word 5,$^)


# bn8
build/bn8_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN8 -DINT8_ACT -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DSCALAR -DBN8 -DINT8_ACT -c $< -o ${@F}
endif

build/bn8_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN8 -DSCALAR -DSTRIDE1 -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBN8 -DSCALAR -DSTRIDE1 -c $< -o ${@F}
endif

build/bn8_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN8 -DSCALAR -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBN8 -DSCALAR  -c $< -o ${@F}
endif

build/bn8_conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN8 -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DSCALAR -DBN8 -c $< -o ${@F}
endif

build/bn8_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a: build/bn8_conv2dk1_fused_relu.o build/bn8_conv2dk3_dw_stride1.o build/bn8_conv2dk1_i8.o 
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)

# bn9
build/bn9_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN9 -DINT8_ACT -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS}  -DSCALAR -DBN9 -DINT8_ACT -c $< -o ${@F}
endif

build/bn9_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN9 -DSCALAR -DSTRIDE1 -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBN9 -DSCALAR -DSTRIDE1 -c $< -o ${@F}
endif

build/bn9_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN9 -DSCALAR -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBN9 -DSCALAR -c $< -o ${@F}
endif

build/bn9_combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a: build/bn9_conv2dk1_fused_relu.o build/bn9_conv2dk3_dw_stride1.o build/bn9_conv2dk1_skip.o 
	mkdir -p ${@D}
	ar rvs $@ $< $(word 2,$^) $(word 3,$^)
# ****************************************************************************************

build/combined_bn_8_9.a: build/bn8_conv2dk1_fused_relu.o build/bn8_conv2dk3_dw_stride1.o build/bn8_conv2dk1_skip.o build/bn9_conv2dk1_fused_relu.o build/bn9_conv2dk3_dw_stride1.o build/bn9_conv2dk1_skip.o
	mkdir -p ${@D}
	ar rvs $@ $^

# ****************************************************************************************

# build/final_%.xclbin build/inst_%.bin &: build/aie2_%.mlir build/combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a build/combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a build/combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a build/combined_conv2dk3dwstride1_conv2dk1skip.a build/combined_bn0_bn1.a  build/combined_bn2_bn3.a
build/final_%.xclbin build/inst_%.bin &: build/aie2_%.mlir build/combined_con2dk1fusedrelu_conv2dk3dwstride2_conv2dk1.a build/combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1.a build/combined_con2dk1fusedrelu_conv2dk3dwstride1_conv2dk1skip.a build/combined_conv2dk3dwstride1_conv2dk1skip.a build/combined_bn_8_9.a 

	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts_$*.bin ${<F}
else
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
			--aie-generate-npu-insts --npu-insts-name=insts_$*.bin $(<F)
endif

shallow_clean:
	rm -rf build/*.elf* build/*.lst build/*.bif build/*.mlir.prj build/*.mlir log build/.xclbin build/sim build/*.bin 

clean:
	rm -rf build __pycache__

run_py: build/final_bn_chain.xclbin build/insts_bn_chain.bin build/aie2_bn_chain.mlir
	${powershell} python3 ${srcdir}/test_bn_chain.py -x build/final_bn_chain.xclbin -i build/insts_bn_chain.bin -k MLIR_AIE

run_py_0: build/final_bn_0.xclbin build/insts_bn_0.bin build/aie2_bn_0.mlir
	${powershell} python3 ${srcdir}/test_bn_0.py -x build/final_bn_0.xclbin -i build/insts_bn_0.bin -k MLIR_AIE

run_py_1: build/final_bn_1.xclbin build/insts_bn_1.bin build/aie2_bn_1.mlir
	${powershell} python3 ${srcdir}/test_bn_1.py -x build/final_bn_1.xclbin -i build/insts_bn_1.bin -k MLIR_AIE

run_py_2: build/final_bn_2.xclbin build/insts_bn_2.bin build/aie2_bn_2.mlir
	${powershell} python3 ${srcdir}/test_bn_2.py -x build/final_bn_2.xclbin -i build/insts_bn_2.bin -k MLIR_AIE

run_py_3: build/final_bn_3.xclbin build/insts_bn_3.bin build/aie2_bn_3.mlir
	${powershell} python3 ${srcdir}/test_bn_3.py -x build/final_bn_3.xclbin -i build/insts_bn_3.bin -k MLIR_AIE

run_py_4: build/final_bn_4.xclbin build/insts_bn_4.bin build/aie2_bn_4.mlir
	${powershell} python3 ${srcdir}/test_bn_4.py -x build/final_bn_4.xclbin -i build/insts_bn_4.bin -k MLIR_AIE

run_py_5: build/final_bn_5.xclbin build/insts_bn_5.bin build/aie2_bn_5.mlir
	${powershell} python3 ${srcdir}/test_bn_5.py -x build/final_bn_5.xclbin -i build/insts_bn_5.bin -k MLIR_AIE

run_py_6: build/final_bn_6.xclbin build/insts_bn_6.bin build/aie2_bn_6.mlir
	${powershell} python3 ${srcdir}/test_bn_6.py -x build/final_bn_6.xclbin -i build/insts_bn_6.bin -k MLIR_AIE


run_py_7: build/final_bn_7.xclbin build/insts_bn_7.bin build/aie2_bn_7.mlir
	${powershell} python3 ${srcdir}/test_bn_7.py -x build/final_bn_7.xclbin -i build/insts_bn_7.bin -k MLIR_AIE

run_py_8: build/final_bn_8.xclbin build/insts_bn_8.bin build/aie2_bn_8.mlir
	${powershell} python3 ${srcdir}/test_bn_8.py -x build/final_bn_8.xclbin -i build/insts_bn_8.bin -k MLIR_AIE

run_py_9: build/final_bn_9.xclbin build/insts_bn_9.bin build/aie2_bn_9.mlir
	${powershell} python3 ${srcdir}/test_bn_9.py -x build/final_bn_9.xclbin -i build/insts_bn_9.bin -k MLIR_AIE

#run_py_0_1: build/final_bn_0_1.xclbin build/insts_bn_0_1.bin build/aie2_bn_0_1.mlir
#	${powershell} python3 ${srcdir}/test_bn_0_1.py -x build/final_bn_0_1.xclbin -i build/insts_bn_0_1.bin -k MLIR_AIE


#run_py_2_3: build/final_bn_2_3.xclbin build/insts_bn_2_3.bin build/aie2_bn_2_3.mlir
#	${powershell} python3 ${srcdir}/test_bn_2_3.py -x build/final_bn_2_3.xclbin -i build/insts_bn_2_3.bin -k MLIR_AIE

run_py_8_9: build/final_bn_8_9.xclbin build/insts_bn_8_9.bin build/aie2_bn_8_9.mlir build/combined_bn_8_9.a
	${powershell} python3 ${srcdir}/test_bn_8_9.py -x build/final_bn_8_9.xclbin -i build/insts_bn_8_9.bin -k MLIR_AIE
