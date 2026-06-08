#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../makefile-common

PYTHONPATH := $(PYTHONPATH):$(realpath $(srcdir)/..)
export PYTHONPATH

MLIR_FILE = aie2_mobilenet_iron
TEST_FILE = test_mobilenet

all: build/final_mobilenet.xclbin run_py

# objs: alias for the xclbin build, used by run_e2e.lit to bootstrap the
# per-kernel .o files.  Under @iron.jit those .o files land inside
# build/final_mobilenet.prj/ (the per-xclbin scratch dir), so run_e2e.sh
# symlinks from there.
.PHONY: objs
objs: build/final_mobilenet.xclbin

# aie2_mobilenet_iron is @iron.jit; the design's compile path builds
# every .o file itself (via the kernels.bn_* library) and invokes aiecc
# directly with the right flags (incl. --dynamic-objFifos=false, set on
# the decorator).
build/final_mobilenet.xclbin:
	mkdir -p $(dir $@)
	python3 -m mobilenet.${MLIR_FILE} \
		--xclbin-path $@ --insts-path build/insts.bin -d ${devicename}

build/${MLIR_FILE}.mlir: ${srcdir}/${MLIR_FILE}.py
	mkdir -p ${@D}
	python3 -m mobilenet.${MLIR_FILE} --emit-mlir -d ${devicename} > $@

run_py: build/final_mobilenet.xclbin
	${powershell} python3 -m mobilenet.${TEST_FILE} -x build/final_mobilenet.xclbin -i build/insts.bin -k MLIR_AIE

shallow_clean:
	rm -rf build/*.elf* build/*.lst build/*.bif build/*.mlir.prj build/*.mlir log build/.xclbin build/sim build/*.bin

clean:
	rm -rf build __pycache__ log
