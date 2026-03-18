#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.bin for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../../makefile-common
VPATH := ${srcdir}/../../../../aie_kernels/aie2

mlirFileName = aie2_bottleneckC
testSource = test_bottleneckC

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
ifeq (${devicename}, npu)
	LOCAL_CHESS_FLAGS := ${CHESSCCWRAP2_FLAGS}
	LOCAL_PEANO_FLAGS := ${PEANOWRAP2_FLAGS}
else
	LOCAL_CHESS_FLAGS := ${CHESSCCWRAP2P_FLAGS}
	LOCAL_PEANO_FLAGS := ${PEANOWRAP2P_FLAGS}
endif 

CHESS ?= false
colshift ?= $(if $(filter npu,$(devicename)),1,0)

trace_size = 32768
# DUMMY = _dummy
DUMMY =

OBJ = build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o \
	build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o build/bn14_1_conv2dk1_put.o \
	build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o \
	build/bn14_conv2dk1_skip_get.o 

all: $(OBJ)  build/final.xclbin run_py
trace: $(OBJ)  build/trace.xclbin trace_py

newFlag=_NEW

build/${mlirFileName}.mlir: ${srcdir}/${mlirFileName}.py
	mkdir -p ${@D}
	python3 $< > $@

build/${mlirFileName}_trace.mlir: ${srcdir}/${mlirFileName}.py
	mkdir -p ${@D}
	python3 $< -t ${trace_size} > $@

# ************************************** bottlneck 13 **************************************
build/bn13_1_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	echo $(new) World with ${newFlag}
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS} -DBN13_1_PARTIAL_PUT_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DBN13_1_PARTIAL_PUT_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}
endif 

build/bn13_1_conv2dk1_get.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	echo $(new) World with ${newFlag}
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS} -DBN13_1_PARTIAL_GET_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DBN13_1_PARTIAL_GET_I8_CAS_WIDTH${newFlag}  -c $< -o ${@F}
endif 

build/bn13_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	echo $(new) World with ${newFlag}
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS} -DSCALAR -DBN13 -DSTRIDE1_OUT_SPLIT -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DSCALAR -DBN13 -DSTRIDE1_OUT_SPLIT -c $< -o ${@F}
endif 

build/bn13_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS} -DBN13_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH${newFlag}  -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DBN13_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH${newFlag} -c $< -o ${@F}
endif 


build/bn13_conv2dk1_skip_get.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS} -DBN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH${newFlag}  -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DBN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}
endif 

# ************************************** bottlneck 14 **************************************
build/bn14_1_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS} -DBN14_1_PARTIAL_PUT_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DBN14_1_PARTIAL_PUT_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}
endif 

build/bn14_1_conv2dk1_get.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS} -DBN14_1_PARTIAL_GET_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DBN14_1_PARTIAL_GET_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}
endif 

build/bn14_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS} -DSCALAR -DBN14 -DSTRIDE1_OUT_SPLIT -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DSCALAR -DBN14 -DSTRIDE1_OUT_SPLIT -c $< -o ${@F}
endif 

build/bn14_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS} -DBN14_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH${newFlag} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DBN14_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH${newFlag} -c $< -o ${@F}
endif 

build/bn14_conv2dk1_skip_get.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${LOCAL_CHESS_FLAGS} -DBN14_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${LOCAL_PEANO_FLAGS} -DBN14_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}
endif 

# ************************************** bottleneck B **************************************
build/final.xclbin build/insts.bin &: build/${mlirFileName}.mlir $(OBJ)
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<F)
else
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--no-xchesscc --no-xbridge \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<F)
endif

build/trace.xclbin build/insts_trace.bin &: build/${mlirFileName}_trace.mlir $(OBJ)
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts_trace.bin $(<F)
else
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--no-xchesscc --no-xbridge \
		--aie-generate-npu-insts --npu-insts-name=insts_trace.bin $(<F)
endif

shallow_clean:
	rm -rf build/*.elf* build/*.lst build/*.bif build/*.mlir.prj build/*.mlir log build/.xclbin build/sim build/*.bin 

clean:
	rm -rf build __pycache__


run_py: build/final.xclbin build/${mlirFileName}.mlir
	${powershell} python3 ${srcdir}/${testSource}.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE

trace_py: build/trace.xclbin build/${mlirFileName}_trace.mlir
	${powershell} python3 ${srcdir}/${testSource}.py -x build/trace.xclbin -i build/insts_trace.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../utils/parse_trace.py --filename trace.bin --mlir build/${mlirFileName}_trace.mlir --colshift ${colshift} > trace_mbv4_C.json
