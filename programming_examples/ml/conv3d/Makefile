##===- Makefile -----------------------------------------------------------===##
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

devicename ?= npu
targetname = conv3d

CHESS ?= false

# Design parameters
depth = 8
height = 8
width = 8
in_channels = 8
out_channels = 8
vectorized ?= false  # Start with scalar, set to true for vectorized
trace_size = 0

VPATH := ${srcdir}/../../../aie_kernels/aie2

# Compilation flags
ifeq ($(vectorized),true)
VECTORIZED_FLAGS = -DUINT8_ACT
else
VECTORIZED_FLAGS = -DSCALAR -DUINT8_ACT
endif

all: build/conv3dk3_ui8.o build/final.xclbin

build/conv3dk3_ui8.o: conv3dk3.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} ${VECTORIZED_FLAGS} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2_FLAGS} ${VECTORIZED_FLAGS} -c $< -o ${@F}
endif

build/aie2.mlir: ${srcdir}/conv3d.py
	mkdir -p ${@D}
	python3 $< ${devicename} ${depth} ${height} ${width} ${in_channels} ${out_channels} ${trace_size} > $@

build/final.xclbin: build/aie2.mlir build/conv3dk3_ui8.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts \
		--no-compile-host --no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

run_py: build/final.xclbin
	python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE \
		-d ${depth} -ht ${height} -wd ${width} -ic ${in_channels} -oc ${out_channels}

trace_py: build/final.xclbin
	python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE \
		-d ${depth} -ht ${height} -wd ${width} -ic ${in_channels} -oc ${out_channels} \
		--trace_sz ${trace_size}

clean:
	rm -rf build log *.elf *.xclbin *.bin *.mlir.prj
