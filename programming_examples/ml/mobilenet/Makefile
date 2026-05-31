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

all: build/final_mobilenet.xclbin run_py
trace: build/trace_mobilenet.xclbin trace_py

MLIR_FILE=aie_mobilenet
TEST_FILE=test_mobilenet

PYTHONPATH := $(PYTHONPATH):$(shell cd .. && pwd)
export PYTHONPATH

trace_size = 32768
# DUMMY = _dummy
DUMMY =

new= True
newFlag=_NEW

MLIR_FILE = aie2_mobilenet_iron
TEST_FILE = test_mobilenet
OBJ = \
	build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o \
	build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o \
	build/post_conv2dk1_relu_xy_pool_padded_i8_ui8.o \
	build/post_L2_conv2dk1_relu_ui16_ui16_pad.o

# Build only the kernel .o files (no MLIR/xclbin). Used by run_e2e.lit to
# bootstrap kernels on a fresh CI checkout where the main mobilenet hasn't
# been built yet.
.PHONY: objs
objs: $(OBJ)


build/${MLIR_FILE}.mlir: ${srcdir}/${MLIR_FILE}.py
	mkdir -p ${@D}
	python3 $< > $@

# build/${MLIR_FILE}_trace.mlir: ${srcdir}/${MLIR_FILE}_trace.py 
build/${MLIR_FILE}_trace.mlir: ${srcdir}/${MLIR_FILE}.py
	mkdir -p ${@D}
	python3 $< -t ${trace_size} > $@

# init (3x3 stride-2) and bn0..bn12 are built via the iron kernel library;
# init.py:init_conv calls kernels.bn_conv2dk3, regular.py + pipeline.py use
# kernels.bn_conv2dk3_dw / bn_conv2dk1_relu / bn_conv2dk1_i8 / bn_conv2dk1_skip.
# aiecc compiles the .o files as part of the @iron.jit pipeline.

# ************************************** bottlneck 13 **************************************
build/bn13_1_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
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

# mobilenet_iron is @iron.jit, so the design's compile path builds the
# kernels.X-managed .o files itself and invokes aiecc directly with the
# right flags (incl. --dynamic-objFifos=false, set on the decorator).
# The non-library kernels (bn1..bn14 + post + cascade — see $(OBJ)) still
# come from the Makefile rules below and land in build/ alongside the
# library-built .o files; aiecc picks up both.
build/final_mobilenet.xclbin: $(OBJ)
	mkdir -p ${@D}
	python3 ${srcdir}/${MLIR_FILE}.py \
		--xclbin-path $@ --insts-path build/insts.bin -d ${devicename}

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

run_py: build/final_mobilenet.xclbin build/${MLIR_FILE}.mlir
	${powershell} python3 ${srcdir}/${TEST_FILE}.py -x build/final_mobilenet.xclbin -i build/insts.bin -k MLIR_AIE

trace_py: build/trace_mobilenet.xclbin build/${MLIR_FILE}_trace.mlir
	${powershell} python3 ${srcdir}/${TEST_FILE}.py -x build/trace_mobilenet.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../utils/parse_trace.py --filename trace.txt --mlir build/${MLIR_FILE}_trace.mlir --colshift 1 > trace_mbv4_full.json

shallow_clean:
	rm -rf build/*.elf* build/*.lst build/*.bif build/*.mlir.prj build/*.mlir log build/.xclbin build/sim build/*.bin 

clean:
	rm -rf build __pycache__ log
