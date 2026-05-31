#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ${COMPILE_WRAPPER}
# Copyright (C) 2024, Advanced Micro Devices, Inc.
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../makefile-common
VPATH := ${srcdir}/../../../aie_kernels/aie2

ifeq (${devicename}, npu)
	LOCAL_CHESS_FLAGS := ${CHESSCCWRAP2_FLAGS}
	LOCAL_PEANO_FLAGS := ${PEANOWRAP2_FLAGS}
else
	LOCAL_CHESS_FLAGS := ${CHESSCCWRAP2P_FLAGS}
	LOCAL_PEANO_FLAGS := ${PEANOWRAP2P_FLAGS}
endif 

CHESS ?= false

ifeq ($(CHESS), true)
	COMPILE_WRAPPER := xchesscc_wrapper ${LOCAL_CHESS_FLAGS}
else
	COMPILE_WRAPPER := ${PEANO_INSTALL_DIR}/bin/clang++ ${LOCAL_PEANO_FLAGS}
endif

colshift ?= $(if $(filter npu,$(devicename)),1,0)

all: build/final_mobilenet.xclbin run_py
trace: build/trace_mobilenet.xclbin trace_py

MLIR_FILE=aie_mobilenet
TEST_FILE=test_mobilenet

PYTHONPATH := $(PYTHONPATH):$(shell cd .. && pwd)
export PYTHONPATH

trace_size = 32768
# DUMMY = _dummy
DUMMY =

new= True
newFlag=_NEW

MLIR_FILE = aie2_mobilenet_iron
TEST_FILE = test_mobilenet

build/${MLIR_FILE}.mlir: ${srcdir}/${MLIR_FILE}.py
	mkdir -p ${@D}
	python3 -m mobilenet.${MLIR_FILE} --emit-mlir -d ${devicename} > $@

# build/${MLIR_FILE}_trace.mlir: ${srcdir}/${MLIR_FILE}_trace.py
build/${MLIR_FILE}_trace.mlir: ${srcdir}/${MLIR_FILE}.py
	mkdir -p ${@D}
	python3 -m mobilenet.${MLIR_FILE} --emit-mlir -d ${devicename} -t ${trace_size} > $@

# Every mobilenet kernel (init, bn0..bn14, post_l1, post_l2) is built via
# the iron kernel library; bottleneck/*.py calls kernels.bn_*. aiecc
# compiles all .o files as part of the @iron.jit pipeline.

# ************************************** mobilenet ********************************************

# mobilenet_iron is @iron.jit; the design's compile path builds every
# .o file itself and invokes aiecc directly with the right flags (incl.
# --dynamic-objFifos=false, set on the decorator).
build/final_mobilenet.xclbin:
	mkdir -p $(dir $@)
	python3 -m mobilenet.${MLIR_FILE} \
		--xclbin-path $@ --insts-path build/insts.bin -d ${devicename}

build/trace_mobilenet.xclbin: build/${MLIR_FILE}_trace.mlir $(OBJ)
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
		--aie-generate-npu-insts --npu-insts-name=insts.bin ${<F}
else
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
   		--no-xchesscc --no-xbridge \
		--aie-generate-npu-insts --npu-insts-name=insts.bin ${<F}
endif

run_py: build/final_mobilenet.xclbin build/${MLIR_FILE}.mlir
	${powershell} python3 -m mobilenet.${TEST_FILE} -x build/final_mobilenet.xclbin -i build/insts.bin -k MLIR_AIE

trace_py: build/trace_mobilenet.xclbin build/${MLIR_FILE}_trace.mlir
	${powershell} python3 -m mobilenet.${TEST_FILE} -x build/trace_mobilenet.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../utils/parse_trace.py --filename trace.txt --mlir build/${MLIR_FILE}_trace.mlir --colshift 1 > trace_mbv4_full.json

shallow_clean:
	rm -rf build/*.elf* build/*.lst build/*.bif build/*.mlir.prj build/*.mlir log build/.xclbin build/sim build/*.bin 

clean:
	rm -rf build __pycache__ log
