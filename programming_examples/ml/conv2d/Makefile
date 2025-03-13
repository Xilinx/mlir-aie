#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

device ?= $(if $(filter 1,$(NPU2)),npu2,npu)
mlirFileName = aie

# Modify these params to configure design
width = 32
height = 32
in_channels = 64
out_channels = 64
vectorized ?= true
trace_size = 16384

ifeq ($(device),npu2)
VPATH := ${srcdir}/../../../aie_kernels/aie2p
else
VPATH := ${srcdir}/../../../aie_kernels/aie2
endif


aie_py_src = conv2d.py
aie_py_trace_src = conv2d_alt.py
use_alt ?= 0

ifeq (${use_alt}, 1)
aie_py_src=conv2d_alt.py
endif

all: build/conv2dk1_i8.o build/final.xclbin

build/conv2dk1_i8.o: conv2dk1_i8.cc
	mkdir -p ${@D}
ifeq ($(device),npu)
ifeq ($(vectorized), true)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2_FLAGS} -DINT8_ACT -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2_FLAGS} -DSCALAR -DINT8_ACT -c $< -o ${@F}
endif
else ifeq ($(device),npu2)
ifeq ($(vectorized), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DINT8_ACT -c $< -o ${@F}
else
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DSCALAR -DINT8_ACT -c $< -o ${@F}
endif
else
	echo "Device type not supported"
endif

build/${mlirFileName}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${device} ${width} ${height} ${in_channels} ${out_channels} 0 > $@

build/${mlirFileName}_trace.mlir: ${srcdir}/${aie_py_trace_src}
	mkdir -p ${@D}
	python3 $< ${device} ${width} ${height} ${in_channels} ${out_channels} ${trace_size} > $@

build/final.xclbin: build/${mlirFileName}.mlir build/conv2dk1_i8.o 
	mkdir -p ${@D} 
ifeq ($(device),npu)
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.txt $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--xclbin-name=${@F} --npu-insts-name=insts.txt ${<F}
endif

build/final_trace.xclbin: build/${mlirFileName}_trace.mlir build/conv2dk1_i8.o 
	mkdir -p ${@D} 
ifeq ($(device),npu)
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts_trace.txt $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--xclbin-name=${@F} --npu-insts-name=insts_trace.txt ${<F}
endif

run_py: build/final.xclbin
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.txt -k MLIR_AIE -wd ${width} -ht ${height} -ic ${in_channels} -oc ${out_channels}

trace_py: build/final_trace.xclbin
	${powershell} python3 ${srcdir}/test.py -x build/final_trace.xclbin -i build/insts_trace.txt -k MLIR_AIE -wd ${width} -ht ${height} -ic ${in_channels} -oc ${out_channels} -t ${trace_size}
	${srcdir}/../../utils/parse_trace.py --filename log/trace_conv2d.txt --mlir build/aie_trace.mlir --colshift 1 > log/trace_conv2d.json
	${srcdir}/../../utils/get_trace_summary.py --filename log/trace_conv2d.json

clean:
	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj log* *.xclbin sim \
		chess* *.o insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe
