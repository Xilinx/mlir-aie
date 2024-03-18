##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../makefile-common

ACDC_AIE = $(dir $(shell which aie-opt))/..

SHELL := /bin/bash

all: build/aie.mlir vck5000 run

scale.o: scale.cc
	xchesscc -p me -P ${VITIS_ROOT}/aietools/data/versal_prod/lib/ -L ${VITIS_ROOT}/aietools/data/versal_prod/lib/ -c ./scale.cc

build/aie.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< > $@

vck5000: build/aie.mlir scale.o
	aiecc.py --link_against_hsa --host-target=x86_64-amd-linux-gnu build/aie.mlir \
						-I/opt/xaiengine/include \
						-I$(ACDC_AIE)/runtime_lib/x86_64-hsa/test_lib/include \
            -I${ROCM_ROOT}/include \
						-L/opt/xaiengine/lib \
            -L/lib/x86_64-linux-gnu/ \
						test.cpp \
						$(ACDC_AIE)/runtime_lib/x86_64-hsa/test_lib/src/test_library.cpp \
            ${ROCM_ROOT}/lib/libhsa-runtime64.so.1.9.0 \
						-Wl,-R/opt/xaiengine/lib \
            -Wl,-rpath,${ROCM_ROOT}/lib \
						-Wl,--whole-archive -Wl,--no-whole-archive -lstdc++ -ldl -lelf -o test.elf

run: test.elf
	./test.elf

clean:
	rm -rf build _build inst aie.mlir.prj core_* test.elf *.o
