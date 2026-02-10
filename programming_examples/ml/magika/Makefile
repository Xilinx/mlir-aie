#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

targetname = group0
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

mlirFileName = aie

aie_py_src=${targetname}.py
use_placed?=1
CHESS ?= false
trace_size = 16384
colshift ?= $(if $(filter npu,$(devicename)),1,0)

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

#ifeq (${targetname}, group2)
#alloc_scheme=--alloc-scheme=basic-sequential
#else
#alloc_scheme=--alloc-scheme=bank-aware
#endif

alloc_scheme=--alloc-scheme=basic-sequential

VPATH := ${srcdir}/kernels

all: build/aie.mlir

build/${mlirFileName}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} > $@

build/${mlirFileName}_trace.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -t ${trace_size} > $@

build/%.o: %.cc
	mkdir -p ${@D}
ifeq (${devicename}, npu)
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}; 
else 
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
endif
else ifeq (${devicename}, npu2)
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -c $< -o ${@F}; 
else 
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
endif
else
	echo "Device type not supported"
endif

build/group0a.o: group0.cc
	mkdir -p ${@D}
ifeq (${devicename}, npu)
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DGROUPA -c $< -o ${@F}; 
else 
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2_FLAGS} -DGROUPA -c $< -o ${@F}
endif
else ifeq (${devicename}, npu2)
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DGROUPA -c $< -o ${@F}; 
else 
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2P_FLAGS} -DGROUPA -c $< -o ${@F}
endif
else
	echo "Device type not supported"
endif

build/group0b.o: group0.cc
	mkdir -p ${@D}
ifeq (${devicename}, npu)
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DGROUPB -c $< -o ${@F}; 
else 
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2_FLAGS} -DGROUPB -c $< -o ${@F}
endif
else ifeq (${devicename}, npu2)
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DGROUPB -c $< -o ${@F}; 
else 
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2P_FLAGS} -DGROUPB -c $< -o ${@F}
endif
else
	echo "Device type not supported"
endif

#build/final.xclbin: build/${mlirFileName}.mlir build/group0a.o build/group0b.o build/group1.o build/group2.o  
build/final.xclbin: build/${mlirFileName}.mlir build/group0a.o build/group0b.o build/group2.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} ${alloc_scheme} \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)
else
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} ${alloc_scheme} \
    		--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%) 
endif

build/final_trace.xclbin: build/${mlirFileName}_trace.mlir build/group0a.o build/group0b.o build/group2.o
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} ${alloc_scheme} \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)
else
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} ${alloc_scheme} \
    		--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%) 
endif

run_py: build/final.xclbin
	#${powershell} python3 ${srcdir}/test.py -v 1 -x build/final.xclbin -i build/insts.bin -k MLIR_AIE
	${powershell} python3 ${srcdir}/test_${targetname}.py -v 1 -x build/final.xclbin -i build/insts.bin -k MLIR_AIE

trace_py: build/final_trace.xclbin
	${powershell} python3 ${srcdir}/test_${targetname}.py -v 1 -x build/final_trace.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../python/utils/trace/parse.py --input log/trace_magika.txt --mlir build/${mlirFileName}_trace.mlir --output trace_${targetname}.json
	${srcdir}/../../../python/utils/trace/get_trace_summary.py --input trace_${targetname}.json

clean:
	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj log .xclbin sim \
		chess* *.o insts.bin \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe
