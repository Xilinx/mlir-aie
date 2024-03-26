# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

ACDC_AIE = $(dir $(shell which aie-opt))/..

all: kernel.o test.elf

kernel.o: kernel.cc kernel.h
	xchesscc -p me -P ${VITIS_ROOT}/aietools/data/versal_prod/lib/ -L ${VITIS_ROOT}/aietools/data/versal_prod/lib/ -c ./kernel.cc

test.elf: aie.mlir test.cpp
	aiecc.py -v --link_against_hsa --host-target=x86_64-amd-linux-gnu --sysroot= $< \
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
						-Wl,--whole-archive -Wl,--no-whole-archive -lstdc++ -ldl -lelf -o $@

clean::
	rm -rf acdc_project *.elf core* aie.mlir.prj *.o
