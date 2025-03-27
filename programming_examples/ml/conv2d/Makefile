#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

device ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = conv2d
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

colshift ?= $(if $(filter 1,$(NPU2)),0,1)
aie_py_src = ${targetname}.py
aie_py_placed_src = ${targetname}_placed.py
use_placed ?= 0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
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
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2P_FLAGS} -DINT8_ACT -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2P_FLAGS} -DSCALAR -DINT8_ACT -c $< -o ${@F}
endif
else
	echo "Device type not supported"
endif

build/aie2.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${device} ${width} ${height} ${in_channels} ${out_channels} 0 > $@

build/aie2_trace.mlir: ${srcdir}/${aie_py_placed_src}
	mkdir -p ${@D}
	python3 $< ${device} ${width} ${height} ${in_channels} ${out_channels} ${trace_size} > $@

build/final.xclbin: build/aie2.mlir build/conv2dk1_i8.o 
	mkdir -p ${@D} 
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

build/final_trace.xclbin: build/aie2_trace.mlir build/conv2dk1_i8.o 
	mkdir -p ${@D} 
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts_trace.bin $(<:%=../%)

run_py: build/final.xclbin
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -wd ${width} -ht ${height} -ic ${in_channels} -oc ${out_channels}

trace_py: build/final_trace.xclbin
	${powershell} python3 ${srcdir}/test.py -x build/final_trace.xclbin -i build/insts_trace.bin -k MLIR_AIE -wd ${width} -ht ${height} -ic ${in_channels} -oc ${out_channels} -t ${trace_size}
	${srcdir}/../../utils/parse_trace.py --filename log/trace_conv2d.txt --mlir build/aie2_trace.mlir --colshift ${colshift} > log/trace_conv2d.json
	${srcdir}/../../utils/get_trace_summary.py --filename log/trace_conv2d.json

clean:
	rm -rf build *.elf* *.lst *.bif aie2.mlir.prj log* *.xclbin sim \
		chess* *.o insts.bin \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe
