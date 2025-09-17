#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = conv2dk14
mlirFileName = aie

width = 896
height = 896
kernel_size = 14
in_channels = 4
out_channels = 1152
trace_size = 32768
CHESS ?= false
vectorized ?= true

ifeq ($(devicename),npu2)
VPATH := ${srcdir}/../../../aie_kernels/aie2p
else
VPATH := ${srcdir}/../../../aie_kernels/aie2
endif

ifeq ($(vectorized),true)
VECTORIZED_FLAGS = -DUINT8_ACT
else
VECTORIZED_FLAGS = -DSCALAR -DUINT8_ACT
endif

aie_py_src = ${targetname}.py
aie_py_placed_src = ${targetname}_placed.py
use_placed ?= 1

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

all: build/conv2dk14.o build/final.xclbin

build/conv2dk14.o: conv2dk14.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} ${VECTORIZED_FLAGS} -c $< -o ${@F};
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2_FLAGS} ${VECTORIZED_FLAGS} -c $< -o ${@F}
endif
else ifeq ($(devicename),npu2)
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} ${VECTORIZED_FLAGS} -c $< -o ${@F};
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2P_FLAGS} ${VECTORIZED_FLAGS} -c $< -o ${@F}
endif
else
	echo "Device type not supported"
endif

build/aie2.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} ${width} ${height} ${in_channels} ${out_channels} ${kernel_size} 0 > $@

build/aie2_trace.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} ${width} ${height} ${in_channels} ${out_channels} ${kernel_size} ${trace_size} > $@

build/final.xclbin: build/aie2.mlir build/conv2dk14.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		 --xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge --xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)
endif

build/final_trace.xclbin: build/aie2_trace.mlir build/conv2dk14.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--xclbin-name=${@F} --npu-insts-name=insts_trace.bin $(<:%=../%)
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
		--no-xchesscc --no-xbridge --xclbin-name=${@F} --npu-insts-name=insts_trace.bin $(<:%=../%)
endif

run_py: build/final.xclbin
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -wd ${width} -ht ${height} -ic ${in_channels} -oc ${out_channels} -ksz ${kernel_size}

trace_py: build/final_trace.xclbin
	${powershell} python3 ${srcdir}/test.py -x build/final_trace.xclbin -i build/insts_trace.bin -k MLIR_AIE -wd ${width} -ht ${height} -ic ${in_channels} -oc ${out_channels} -ksz ${kernel_size} -t ${trace_size}
	${srcdir}/../../utils/parse_trace.py --input log/trace_conv2d.txt --mlir build/aie2_trace.mlir --output log/trace_conv2d_14x14.json
	${srcdir}/../../utils/get_trace_summary.py --input log/trace_conv2d_14x14.json

clean:
	rm -rf build *.elf* *.lst *.bif aie2.mlir.prj log* *.xclbin sim \
		chess* *.o insts.bin \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe
