#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.bin for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common

mlirFileName = aie
stride = 1
VPATH := ${srcdir}/../../../../aie_kernels/aie2

# pythonSource=aie2_bn13_layer2_with_output_split

# pythonSource=aie2_bn13bn14
# pythonSource=aie2_bottleneckC
# pythonSource=aie2_bottleneckC_large_wts
pythonSource=aie2_bottleneckC_large_wts_static
#pythonSource=aie2_bottleneckC_large_wts_static_2


# testSource=test_layer2
# testSource=test
testSource=test_static
#testSource=test_static_2

trace_size = 32768
# DUMMY = _dummy
DUMMY =

new= True

newFlag=_NEW


OBJ = build/bn13_1_conv2dk1_put.o build/bn13_1_conv2dk1_get.o build/bn13_conv2dk3_dw.o build/bn13_conv2dk1_put.o build/bn13_conv2dk1_skip_get.o build/bn14_1_conv2dk1_put.o build/bn14_1_conv2dk1_get.o build/bn14_conv2dk3_dw.o build/bn14_conv2dk1_put.o build/bn14_conv2dk1_skip_get.o 


all: $(OBJ)  build/final.xclbin run_py

trace: $(OBJ)  build/trace.xclbin trace_py

# ************************************** bottlneck 13 **************************************

ifdef $(new)
	newFlag=_NEW
else
	newFlag=
endif

newFlag=_NEW




build/bn13_1_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	echo $(new) World with ${newFlag}
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN13_1_PARTIAL_PUT_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

build/bn13_1_conv2dk1_get.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	echo $(new) World with ${newFlag}
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN13_1_PARTIAL_GET_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

build/bn13_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	echo $(new) World with ${newFlag}
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN13 -DSTRIDE1_OUT_SPLIT -c $< -o ${@F}

build/bn13_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN13_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH${newFlag}  -c $< -o ${@F}


build/bn13_conv2dk1_skip_get.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN13_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH${newFlag}  -c $< -o ${@F}

# ************************************** bottlneck 14 **************************************

build/bn14_1_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN14_1_PARTIAL_PUT_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

build/bn14_1_conv2dk1_get.o: bottleneck/bn_conv2dk1_relu${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN14_1_PARTIAL_GET_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

build/bn14_conv2dk3_dw.o: bottleneck/bn_conv2dk3_dw${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DBN14 -DSTRIDE1_OUT_SPLIT -c $< -o ${@F}


build/bn14_conv2dk1_put.o: bottleneck/bn_conv2dk1_i8${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN14_1_INPUT_SPLIT_PARTIAL_PUT_UI8_UI8_CAS_WIDTH${newFlag} -c $< -o ${@F}

build/bn14_conv2dk1_skip_get.o: bottleneck/bn_conv2dk1_skip${DUMMY}.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DBN14_1_INPUT_SPLIT_PARTIAL_GET_UI8_I8_I8_CAS_WIDTH${newFlag} -c $< -o ${@F}

build/${mlirFileName}.mlir: ${srcdir}/${pythonSource}.py
	mkdir -p ${@D}
	python3 $< > $@

build/${mlirFileName}_trace.mlir: ${srcdir}/${pythonSource}.py
	mkdir -p ${@D}
	python3 $< -t ${trace_size} > $@

#insts.bin: build/${mlirFileName}.mlir
#	aiecc.py -v --aie-only-generate-npu --npu-insts-name=$@ $<

build/final.xclbin: build/${mlirFileName}.mlir $(OBJ)
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts_$*.bin $(<F)

build/trace.xclbin: build/${mlirFileName}_trace.mlir $(OBJ)
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts_$*.bin $(<F)


shallow_clean:
	rm -rf build/*.elf* build/*.lst build/*.bif build/*.mlir.prj build/*.mlir log build/.xclbin build/sim build/*.bin 

clean:
	rm -rf build __pycache__


run_py: build/final.xclbin ${srcdir}/${mlirFileName}.py
	${powershell} python3 ${srcdir}/${testSource}.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE

trace_py: build/final.xclbin ${srcdir}/${mlirFileName}.py
	${powershell} python3 ${srcdir}/${testSource}.py -x build/trace.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../utils/parse_trace.py --filename trace.bin --mlir build/${mlirFileName}_trace.mlir --colshift 1 > trace_mbv4_C.json
