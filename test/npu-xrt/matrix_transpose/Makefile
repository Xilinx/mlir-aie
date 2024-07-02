#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

# Required environment variables: 
# - XILINX_VITIS
# - XILINX_XRT
# - MLIR_AIE_INSTALL_DIR or AIECC

AIETOOLS?=${XILINX_VITIS}/aietools
XRT_FLAGS?=-I${XILINX_XRT}/include -L${XILINX_XRT}/lib -lxrt_coreutil
PYTHON?=python3
AIECC?=${MLIR_AIE_INSTALL_DIR}/bin/aiecc.py
RUN_ON_NPU?=
SRC?=.

aie2.mlir: ${SRC}/aie2.py
	python3 ${SRC}/aie2.py > aie2.mlir

kernel.o: ${SRC}/kernel.cc
	xchesscc_wrapper aie2 -I ${AIETOOLS}/include -c ${SRC}/kernel.cc -o kernel.o

final.xclbin insts.txt: aie2.mlir kernel.o
	${PYTHON} ${AIECC} --no-aiesim --aie-generate-cdo --aie-generate-npu --aie-generate-xclbin --no-compile-host --xclbin-name=final.xclbin --npu-insts-name=insts.txt ./aie2.mlir

test: ${SRC}/test.cpp final.xclbin insts.txt
	clang ${SRC}/test.cpp -o test -std=c++17 -Wall ${XRT_FLAGS} -lrt -lstdc++

.PHONY: run
run: test
	${RUN_ON_NPU} ./test

.PHONY: clean
clean:
	-rm -r *.prj *.mlir insts.txt *.o *.lst test *.xclbin
	-rm core_*