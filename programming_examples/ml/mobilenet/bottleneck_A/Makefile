#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../../makefile-common
VPATH := ${srcdir}/../../../../aie_kernels/aie2

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
ifeq (${devicename}, npu)
	LOCAL_CHESS_FLAGS := ${CHESSCCWRAP2_FLAGS}
	LOCAL_PEANO_FLAGS := ${PEANOWRAP2_FLAGS}
else
	LOCAL_CHESS_FLAGS := ${CHESSCCWRAP2P_FLAGS}
	LOCAL_PEANO_FLAGS := ${PEANOWRAP2P_FLAGS}
endif 

CHESS ?= false

ifeq ($(CHESS), true)
	COMPILE_WRAPPER := xchesscc_wrapper ${LOCAL_CHESS_FLAGS}
else
	COMPILE_WRAPPER := ${PEANO_INSTALL_DIR}/bin/clang++ ${LOCAL_PEANO_FLAGS}
endif

colshift ?= $(if $(filter npu,$(devicename)),1,0)

all: build/final_bn_chain.xclbin run_py
trace: build/trace_bn_chain.xclbin trace_py

MLIR_FILE = aie2_bn_0_1_2_3_4_5_6_7_8_9
TEST_FILE = test_bottleneckA

# OBJS_PY = ${srcdir}/aie2_bottleneckA.py ${srcdir}/aie2_bottleneck0.py ${srcdir}/aie2_bottleneck0And1.py ${srcdir}/aie2_bottleneck2And3.py
# OBJS_PY = ${srcdir}/aie2_bottleneckA.py ${srcdir}/aie2_bottleneck0Static.py ${srcdir}/aie2_bottleneck0And1.py ${srcdir}/aie2_bottleneck2And3.py
# OBJS_PY = ${srcdir}/aie2_bottleneckA.py ${srcdir}/aie2_bottleneck0.py ${srcdir}/aie2_bottleneck8And9.py
# OBJS_PY = ${srcdir}/aie2_bottleneckAStatic.py ${srcdir}/aie2_bottleneck0Static.py ${srcdir}/aie2_bottleneck8And9Static.py
OBJS_PY = ${srcdir}/aie2_bottleneckA_subblockStatic.py ${srcdir}/aie2_bottleneckA_subblock0Static.py ${srcdir}/aie2_bottleneckA_subblock_fused2Static.py

# build/aie2_%.mlir: ${srcdir}/aie2_%.py ${OBJS_PY}
build/aie2_%.mlir: ${srcdir}/aie2_%.py
	mkdir -p ${@D}
	python3 $< > $@

build/${MLIR_FILE}.mlir: ${srcdir}/${MLIR_FILE}.py 
	mkdir -p ${@D}
	python3 $< > $@

build/${MLIR_FILE}_trace.mlir: ${srcdir}/${MLIR_FILE}.py 
	mkdir -p ${@D}
	python3 $< -t ${trace_size} > $@

# *********************************** generic bottleneck ***********************************

build/conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR  -DREGULAR -c $< -o ${@F}

build/conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DINT8_ACT -DREGULAR -c $< -o ${@F}


build/conv2dk3_dw_stride2.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DREGULAR  -DSCALAR -DSTRIDE2 -c $< -o ${@F}

build/conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DREGULAR  -DSCALAR  -DSTRIDE1 -c $< -o ${@F}

build/conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DREGULAR -DSCALAR -c $< -o ${@F}

build/conv2dk1_skipui8.o: bottleneck/bn_conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DREGULAR -DSCALAR -DUNSIGNED_SKIP -c $< -o ${@F}


# ************************************** bottleneck 0 **************************************
build/bn0_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN0  -DSCALAR  -DSTRIDE1 -c $< -o ${@F}

build/bn0_conv2dk1_skipui8.o: bottleneck/bn_conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN0 -DSCALAR -DUNSIGNED_SKIP -c $< -o ${@F}


# ************************************** bottleneck 1 **************************************
build/bn1_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN1 -DINT8_ACT -c $< -o ${@F}

build/bn1_conv2dk3_dw_stride2.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN1  -DSCALAR -DSTRIDE2 -c $< -o ${@F}

build/bn1_conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER}  -DSCALAR -DBN1 -c $< -o ${@F}

# ************************************** bottleneck 2 **************************************
build/bn2_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN2 -DINT8_ACT -c $< -o ${@F}

build/bn2_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN2  -DSCALAR  -DSTRIDE1 -c $< -o ${@F}

build/bn2_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN2 -DSCALAR -c $< -o ${@F}

# ************************************** bottleneck 3 **************************************
build/bn3_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN3 -DINT8_ACT -c $< -o ${@F}

build/bn3_conv2dk3_dw_stride2.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN3  -DSCALAR -DSTRIDE2 -c $< -o ${@F}

build/bn3_conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER}  -DSCALAR -DBN3 -c $< -o ${@F}

# ************************************** bottleneck 4 **************************************
build/bn4_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN4 -DINT8_ACT -c $< -o ${@F}

build/bn4_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN4  -DSCALAR  -DSTRIDE1 -c $< -o ${@F}

build/bn4_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN4 -DSCALAR -c $< -o ${@F}

# ************************************** bottleneck 5 **************************************
build/bn5_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN5 -DINT8_ACT -c $< -o ${@F}

build/bn5_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN5  -DSCALAR  -DSTRIDE1 -c $< -o ${@F}

build/bn5_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN5 -DSCALAR -c $< -o ${@F}

build/bn5_conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER}  -DSCALAR -DBN5 -c $< -o ${@F}

# ************************************** bottleneck 6 **************************************
build/bn6_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN6 -DINT8_ACT -c $< -o ${@F}

build/bn6_conv2dk3_dw_stride2.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN6  -DSCALAR -DSTRIDE2 -c $< -o ${@F}

build/bn6_conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER}  -DSCALAR -DBN6 -c $< -o ${@F}

# ************************************** bottleneck 7 **************************************
build/bn7_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN7 -DINT8_ACT -c $< -o ${@F}

build/bn7_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN7  -DSCALAR  -DSTRIDE1 -c $< -o ${@F}

build/bn7_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN7 -DSCALAR -c $< -o ${@F}

# ************************************** bottleneck 8 **************************************
build/bn8_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN8 -DINT8_ACT -c $< -o ${@F}

build/bn8_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN8  -DSCALAR  -DSTRIDE1 -c $< -o ${@F}

build/bn8_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN8 -DSCALAR -c $< -o ${@F}

build/bn8_conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER}  -DSCALAR -DBN8 -c $< -o ${@F}

# ************************************** bottleneck 9 **************************************
build/bn9_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN9 -DINT8_ACT -c $< -o ${@F}

build/bn9_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN9  -DSCALAR  -DSTRIDE1 -c $< -o ${@F}

build/bn9_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN9 -DSCALAR -c $< -o ${@F}

# ************************************** bottleneck A **************************************

OBJ = build/bn0_conv2dk3_dw_stride1.o build/bn0_conv2dk1_skipui8.o \
		build/bn1_conv2dk1_fused_relu.o build/bn1_conv2dk3_dw_stride2.o build/bn1_conv2dk1_i8.o \
		build/bn2_conv2dk1_fused_relu.o build/bn2_conv2dk3_dw_stride1.o build/bn2_conv2dk1_skip.o \
		build/bn3_conv2dk1_fused_relu.o build/bn3_conv2dk3_dw_stride2.o build/bn3_conv2dk1_i8.o \
		build/bn4_conv2dk1_fused_relu.o build/bn4_conv2dk3_dw_stride1.o build/bn4_conv2dk1_skip.o \
		build/bn5_conv2dk1_fused_relu.o build/bn5_conv2dk3_dw_stride1.o build/bn5_conv2dk1_skip.o build/bn5_conv2dk1_i8.o \
		build/bn6_conv2dk1_fused_relu.o build/bn6_conv2dk3_dw_stride2.o build/bn6_conv2dk1_i8.o \
		build/bn7_conv2dk1_fused_relu.o build/bn7_conv2dk3_dw_stride1.o build/bn7_conv2dk1_skip.o \
		build/bn8_conv2dk1_fused_relu.o build/bn8_conv2dk3_dw_stride1.o build/bn8_conv2dk1_skip.o build/bn8_conv2dk1_i8.o \
		build/bn9_conv2dk1_fused_relu.o build/bn9_conv2dk3_dw_stride1.o build/bn9_conv2dk1_skip.o 


build/final_bn_4_5.xclbin build/inst_bn_4_5.bin &: build/aie2_bn_8_9.mlir build/bn8_conv2dk1_fused_relu.o build/bn8_conv2dk3_dw_stride1.o build/bn8_conv2dk1_skip.o build/bn9_conv2dk1_fused_relu.o build/bn9_conv2dk3_dw_stride1.o build/bn9_conv2dk1_skip.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts_bn_4_5.bin ${<F}
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
			--aie-generate-npu-insts --npu-insts-name=insts_bn_4_5.bin $(<F)
endif

build/final_bn_8_9.xclbin build/inst_bn_8_9.bin &: build/aie2_bn_8_9.mlir build/bn8_conv2dk1_fused_relu.o build/bn8_conv2dk3_dw_stride1.o build/bn8_conv2dk1_skip.o build/bn9_conv2dk1_fused_relu.o build/bn9_conv2dk3_dw_stride1.o build/bn9_conv2dk1_skip.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts_bn_8_9.bin ${<F}
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
			--aie-generate-npu-insts --npu-insts-name=insts_bn_8_9.bin $(<F)
endif

build/final_%.xclbin build/inst_%.bin &: build/aie2_%.mlir build/conv2dk1_i8.o build/conv2dk1_fused_relu.o build/conv2dk3_dw_stride2.o build/conv2dk3_dw_stride1.o build/conv2dk1_skip.o build/conv2dk1_skip.o build/conv2dk1_skipui8.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts_$*.bin ${<F}
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
			--aie-generate-npu-insts --npu-insts-name=insts_$*.bin $(<F)
endif

build/final_bn_chain.xclbin: build/${MLIR_FILE}.mlir  $(OBJ)
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts_bn_chain.bin ${<F}
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
		--aie-generate-npu-insts --npu-insts-name=insts_bn_chain.bin ${<F}
endif

build/trace_bn_chain.xclbin: build/${MLIR_FILE}_trace.mlir  $(OBJ)
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts_bn_chain.bin ${<F}
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    		--no-xchesscc --no-xbridge \
		--aie-generate-npu-insts --npu-insts-name=insts_bn_chain.bin ${<F}
endif

# ******************************************************************************************

shallow_clean:
	rm -rf build/*.elf* build/*.lst build/*.bif build/*.mlir.prj build/*.mlir log build/.xclbin build/sim build/*.bin 

clean:
	rm -rf build __pycache__ log

# ******************************************************************************************

run_py: build/final_bn_chain.xclbin build/insts_bn_chain.bin build/${MLIR_FILE}.mlir
	${powershell} python3 ${srcdir}/${TEST_FILE}.py -x build/final_bn_chain.xclbin -i build/insts_bn_chain.bin -k MLIR_AIE

trace_py: build/trace_bn_chain.xclbin build/insts_bn_chain.bin build/${MLIR_FILE}_trace.mlir
	${powershell} python3 ${srcdir}/${TEST_FILE}.py -x build/trace_bn_chain.xclbin -i build/insts_bn_chain.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../utils/parse_trace.py --filename trace.txt --mlir build/${MLIR_FILE}_trace.mlir --colshift ${colshift}  > trace_mbv4_A_chain.json

run_py_0: build/final_bn_0.xclbin build/insts_bn_0.bin build/aie2_bn_0.mlir
	${powershell} python3 ${srcdir}/test_bn_single.py -bn 0 -x build/final_bn_0.xclbin -i build/insts_bn_0.bin -k MLIR_AIE

run_py_1: build/final_bn_1.xclbin build/insts_bn_1.bin build/aie2_bn_1.mlir
	${powershell} python3 ${srcdir}/test_bn_single.py -bn 1 -x build/final_bn_1.xclbin -i build/insts_bn_1.bin -k MLIR_AIE

run_py_2: build/final_bn_2.xclbin build/insts_bn_2.bin build/aie2_bn_2.mlir
	${powershell} python3 ${srcdir}/test_bn_single.py -bn 2 -x build/final_bn_2.xclbin -i build/insts_bn_2.bin -k MLIR_AIE

run_py_3: build/final_bn_3.xclbin build/insts_bn_3.bin build/aie2_bn_3.mlir
	${powershell} python3 ${srcdir}/test_bn_single.py -bn 3 -x build/final_bn_3.xclbin -i build/insts_bn_3.bin -k MLIR_AIE

# run_py_4: build/final_bn_4.xclbin build/insts_bn_4.bin build/aie2_bn_4.mlir
# 	${powershell} python3 ${srcdir}/test_bn_4.py -x build/final_bn_4.xclbin -i build/insts_bn_4.bin -k MLIR_AIE

# run_py_5: build/final_bn_5.xclbin build/insts_bn_5.bin build/aie2_bn_5.mlir
# 	${powershell} python3 ${srcdir}/test_bn_5.py -x build/final_bn_5.xclbin -i build/insts_bn_5.bin -k MLIR_AIE

run_py_6: build/final_bn_6.xclbin build/insts_bn_6.bin build/aie2_bn_6.mlir
	${powershell} python3 ${srcdir}/test_bn_single.py -bn 6 -x build/final_bn_6.xclbin -i build/insts_bn_6.bin -k MLIR_AIE

run_py_7: build/final_bn_7.xclbin build/insts_bn_7.bin build/aie2_bn_7.mlir
	${powershell} python3 ${srcdir}/test_bn_single.py -bn 7 -x build/final_bn_7.xclbin -i build/insts_bn_7.bin -k MLIR_AIE

# run_py_8: build/final_bn_8.xclbin build/insts_bn_8.bin build/aie2_bn_8.mlir
# 	${powershell} python3 ${srcdir}/test_bn_8.py -x build/final_bn_8.xclbin -i build/insts_bn_8.bin -k MLIR_AIE

# run_py_9: build/final_bn_9.xclbin build/insts_bn_9.bin build/aie2_bn_9.mlir
# 	${powershell} python3 ${srcdir}/test_bn_9.py -x build/final_bn_9.xclbin -i build/insts_bn_9.bin -k MLIR_AIE

#run_py_0_1: build/final_bn_0_1.xclbin build/insts_bn_0_1.bin build/aie2_bn_0_1.mlir
#	${powershell} python3 ${srcdir}/test_bn_0_1.py -x build/final_bn_0_1.xclbin -i build/insts_bn_0_1.bin -k MLIR_AIE

#run_py_2_3: build/final_bn_2_3.xclbin build/insts_bn_2_3.bin build/aie2_bn_2_3.mlir
#	${powershell} python3 ${srcdir}/test_bn_2_3.py -x build/final_bn_2_3.xclbin -i build/insts_bn_2_3.bin -k MLIR_AIE

run_py_4_5: build/final_bn_4_5.xclbin build/insts_bn_4_5.bin build/aie2_bn_4_5.mlir
	${powershell} python3 ${srcdir}/test_bn_fused2.py -bn 4_5 -x build/final_bn_4_5.xclbin -i build/insts_bn_4_5.bin -k MLIR_AIE

run_py_8_9: build/final_bn_8_9.xclbin build/insts_bn_8_9.bin build/aie2_bn_8_9.mlir
	${powershell} python3 ${srcdir}/test_bn_fused2.py -bn 8_9 -x build/final_bn_8_9.xclbin -i build/insts_bn_8_9.bin -k MLIR_AIE
