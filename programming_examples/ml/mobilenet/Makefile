#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ${COMPILE_WRAPPER}
# Copyright (C) 2024, Advanced Micro Devices, Inc.
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../makefile-common
VPATH := ${srcdir}/../../../aie_kernels/aie2

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

all: build/final_mobilenet.xclbin run_py_mobilenet
trace: build/trace_mobilenet.xclbin trace_py_mobilenet

MLIR_FILE=aie_mobilenet
TEST_FILE=test_mobilenet

PYTHONPATH := $(PYTHONPATH):$(shell cd .. && pwd)
export PYTHONPATH

trace_size = 32768
# DUMMY = _dummy
DUMMY =

new= True
newFlag=_NEW

MLIR_FILE = aie2_mobilenet
TEST_FILE = test_mobilenet
OBJ = build/init_conv2dk3.o \
	build/bn0_conv2dk3_dw_stride1.o build/bn0_conv2dk1_skipui8.o \
	build/bn1_conv2dk1_fused_relu.o build/bn1_conv2dk3_dw_stride2.o build/bn1_conv2dk1_i8.o \
	build/bn2_conv2dk1_fused_relu.o build/bn2_conv2dk3_dw_stride1.o build/bn2_conv2dk1_skip.o \
	build/bn3_conv2dk1_fused_relu.o build/bn3_conv2dk3_dw_stride2.o build/bn3_conv2dk1_i8.o \
	build/bn4_conv2dk1_fused_relu.o build/bn4_conv2dk3_dw_stride1.o build/bn4_conv2dk1_skip.o \
	build/bn5_conv2dk1_fused_relu.o build/bn5_conv2dk3_dw_stride1.o build/bn5_conv2dk1_skip.o build/bn5_conv2dk1_i8.o \
	build/bn6_conv2dk1_fused_relu.o build/bn6_conv2dk3_dw_stride2.o build/bn6_conv2dk1_i8.o \
	build/bn7_conv2dk1_fused_relu.o build/bn7_conv2dk3_dw_stride1.o build/bn7_conv2dk1_skip.o \
	build/bn8_conv2dk1_fused_relu.o build/bn8_conv2dk3_dw_stride1.o build/bn8_conv2dk1_skip.o build/bn8_conv2dk1_i8.o \
	build/bn9_conv2dk1_fused_relu.o build/bn9_conv2dk3_dw_stride1.o build/bn9_conv2dk1_skip.o \
	build/bn10_conv2dk1_fused_relu.o build/bn10_conv2dk3_dw.o build/bn10_conv2dk1_ui8.o \
	build/bn11_conv2dk1_fused_relu.o build/bn11_conv2dk3_dw.o build/bn11_conv2dk1_skip.o \
	build/bn12_conv2dk1_fused_relu.o build/bn12_conv2dk3_dw_stride2.o build/bn12_conv2dk1_ui8.o \
	build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o \
	build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
	build/post_conv2dk1_relu_xy_pool_padded_i8_ui8.o \
	build/post_L2_conv2dk1_relu_ui16_ui16_pad.o


build/${MLIR_FILE}.mlir: ${srcdir}/${MLIR_FILE}.py 
	mkdir -p ${@D}
	python3 $< > $@

# build/${MLIR_FILE}_trace.mlir: ${srcdir}/${MLIR_FILE}_trace.py 
build/${MLIR_FILE}_trace.mlir: ${srcdir}/${MLIR_FILE}.py 
	mkdir -p ${@D}
	python3 $< -t ${trace_size} > $@

# ************************************** init **********************************************
build/init_conv2dk3.o: bottleneck/bn_conv2dk3${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -c $< -o ${@F}

# ************************************** bottleneck 0+1 ************************************
build/bn0_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN0 -DSCALAR -DSTRIDE1 -c $< -o ${@F}

build/bn0_conv2dk1_skipui8.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN0 -DSCALAR -DUNSIGNED_SKIP -c $< -o ${@F}

build/bn1_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN1 -DINT8_ACT -c $< -o ${@F}

build/bn1_conv2dk3_dw_stride2.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN1 -DSCALAR -DSTRIDE2 -c $< -o ${@F}

build/bn1_conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN1 -c $< -o ${@F}

# ************************************** bottleneck 2 **************************************
build/bn2_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN2 -DINT8_ACT -c $< -o ${@F}

build/bn2_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN2 -DSCALAR -DSTRIDE1 -c $< -o ${@F}

build/bn2_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN2 -DSCALAR -c $< -o ${@F}

# ************************************** bottleneck 3 **************************************
build/bn3_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN3 -DINT8_ACT -c $< -o ${@F}

build/bn3_conv2dk3_dw_stride2.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN3 -DSCALAR -DSTRIDE2 -c $< -o ${@F}

build/bn3_conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN3 -c $< -o ${@F}

# ************************************** bottleneck 4 **************************************
build/bn4_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN4 -DINT8_ACT -c $< -o ${@F}

build/bn4_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN4 -DSCALAR -DSTRIDE1 -c $< -o ${@F}

build/bn4_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN4 -DSCALAR -c $< -o ${@F}

# ************************************** bottleneck 5 **************************************
build/bn5_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN5 -DINT8_ACT -c $< -o ${@F}

build/bn5_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN5 -DSCALAR -DSTRIDE1 -c $< -o ${@F}

build/bn5_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN5 -DSCALAR -c $< -o ${@F}

build/bn5_conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN5 -c $< -o ${@F}

# ************************************** bottleneck 6 **************************************
build/bn6_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN6 -DINT8_ACT -c $< -o ${@F}

build/bn6_conv2dk3_dw_stride2.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN6 -DSCALAR -DSTRIDE2 -c $< -o ${@F}

build/bn6_conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN6 -c $< -o ${@F}

# ************************************** bottleneck 7 **************************************
build/bn7_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN7 -DINT8_ACT -c $< -o ${@F}

build/bn7_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN7 -DSCALAR -DSTRIDE1 -c $< -o ${@F}
build/bn7_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN7 -DSCALAR -c $< -o ${@F}

# ************************************** bottleneck 8 **************************************
build/bn8_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN8 -DINT8_ACT -c $< -o ${@F}

build/bn8_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN8 -DSCALAR -DSTRIDE1 -c $< -o ${@F}

build/bn8_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN8 -DSCALAR -c $< -o ${@F}

build/bn8_conv2dk1_i8.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN8 -c $< -o ${@F}

# ************************************** bottleneck 9 **************************************
build/bn9_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN9 -DINT8_ACT -c $< -o ${@F}

build/bn9_conv2dk3_dw_stride1.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN9 -DSCALAR -DSTRIDE1 -c $< -o ${@F}
build/bn9_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN9 -DSCALAR -c $< -o ${@F}

# ************************************** bottleneck 10 *************************************
build/bn10_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN10 -DINT8_ACT -c $< -o ${@F}

build/bn10_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN10 -DSTRIDE1  -c $< -o ${@F}

build/bn10_conv2dk1_ui8.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN10 -DSCALAR -c $< -o ${@F}

# ************************************** bottleneck 11 *************************************
build/bn11_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN11 -DINT8_ACT -c $< -o ${@F}

build/bn11_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN11 -DSTRIDE1 -c $< -o ${@F}

build/bn11_conv2dk1_skip.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN11 -DSCALAR -c $< -o ${@F}

# ************************************** bottleneck 12 *************************************
build/bn12_conv2dk1_fused_relu.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN12 -DINT8_ACT -c $< -o ${@F}

build/bn12_conv2dk3_dw_stride2.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DSTRIDE2 -DBN12 -c $< -o ${@F}

build/bn12_conv2dk1_ui8.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN12 -DSCALAR -c $< -o ${@F}

# ************************************** bottlneck 13 **************************************
build/bn13_1_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	echo $(new) World with ${newFlag}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN13_1_PARTIAL_PUT_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

build/bn13_1_conv2dk1_get.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	echo $(new) World with ${newFlag}
	cd ${@D} && ${COMPILE_WRAPPER} -DBN13_1_PARTIAL_GET_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

build/bn13_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	echo $(new) World with ${newFlag}
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN13 -DSTRIDE1_OUT_SPLIT -c $< -o ${@F}

build/bn13_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DBN13_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH${newFlag} -c $< -o ${@F}


build/bn13_conv2dk1_skip_get.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DBN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

# ************************************** bottlneck 14 **************************************
build/bn14_1_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DBN14_1_PARTIAL_PUT_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

build/bn14_1_conv2dk1_get.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DBN14_1_PARTIAL_GET_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

build/bn14_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DBN14 -DSTRIDE1_OUT_SPLIT -c $< -o ${@F}


build/bn14_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DBN14_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH${newFlag} -c $< -o ${@F}

build/bn14_conv2dk1_skip_get.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DBN14_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

# ************************************** POST ************************************************
build/post_conv2dk1_relu_xy_pool_i8_ui8.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DCONV_XYPOOL_FUSED_LARGE -DINT8_ACT -c $< -o ${@F}

build/post_L2_conv2dk1_relu_ui16_ui8.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DPOSTL2 -DUINT16_ACT -c $< -o ${@F}

build/post_conv2dk1_relu_xy_pool_padded_i8_ui8.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DCONV_XYPOOL_FUSED_LARGE_PADDED -DINT8_ACT -c $< -o ${@F}

build/post_L2_conv2dk1_relu_ui16_ui16_pad.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	cd ${@D} && ${COMPILE_WRAPPER} -DSCALAR -DPOSTL2_PAD -DUINT16_ACT -c $< -o ${@F}


# ************************************** mobilenet ********************************************

build/final_mobilenet.xclbin: build/${MLIR_FILE}.mlir $(OBJ)
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts.bin ${<F}
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
   		--no-xchesscc --no-xbridge \
		--aie-generate-npu-insts --npu-insts-name=insts.bin ${<F}
endif

build/trace_mobilenet.xclbin: build/${MLIR_FILE}_trace.mlir $(OBJ)
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts.bin ${<F}
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
   		--no-xchesscc --no-xbridge \
		--aie-generate-npu-insts --npu-insts-name=insts.bin ${<F}
endif

run_py_mobilenet: build/final_mobilenet.xclbin build/${MLIR_FILE}.mlir
	${powershell} python3 ${srcdir}/${TEST_FILE}.py -x build/final_mobilenet.xclbin -i build/insts.bin -k MLIR_AIE

trace_py_mobilenet: build/trace_mobilenet.xclbin build/${MLIR_FILE}_trace.mlir
	${powershell} python3 ${srcdir}/${TEST_FILE}.py -x build/trace_mobilenet.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../utils/parse_trace.py --filename trace.txt --mlir build/${MLIR_FILE}_trace.mlir --colshift 1 > trace_mbv4_full.json

shallow_clean:
	rm -rf build/*.elf* build/*.lst build/*.bif build/*.mlir.prj build/*.mlir log build/.xclbin build/sim build/*.bin 

clean:
	rm -rf build __pycache__
