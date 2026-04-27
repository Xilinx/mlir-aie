##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
# G-T3.2-007 worked example: VariableRateFifo end-to-end demo.
# Mirrors the passthrough_kernel Makefile shape; the only
# difference is the kernel object filename and the topology .py.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)
targetname = variable_rate_filter
in_size = 4096   # in bytes (multiple of 64; >= 512)
out_size = 4096  # in bytes (>= in_size for worst-case 100%-forward)
CHESS ?= false

aie_py_src=${targetname}.py

.PHONY: all clean

all: build/final_${in_size}.xclbin

build/filter_first_byte_even.cc.o: filter_first_byte_even.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	echo "Device type not supported"
endif

build/aie_${in_size}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -i1s ${in_size} -os ${out_size} > $@

build/final_${in_size}.xclbin: build/aie_${in_size}.mlir build/filter_first_byte_even.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts_${in_size}.bin \
		$(<:%=../%)

clean:
	rm -rf build _build ${targetname}*.exe
