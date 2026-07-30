##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

targetname = dwconv1d
channels ?= 64
seq_len ?= 128
kernel_size ?= 9
n_cores ?= 8
bias ?= 1

aie_py_src = ${targetname}.py
aieargs = -d ${devicename} -c ${channels} -s ${seq_len} -k ${kernel_size} -n ${n_cores} $(if $(filter 0,${bias}),--no_bias,)

xclbin_target = build/final.xclbin
insts_target  = build/insts.bin
elf_target    = build/insts.elf

.PHONY: all
all: ${xclbin_target} ${insts_target} ${elf_target}

$(eval $(call jit_xclbin_elf,${xclbin_target},${insts_target},${elf_target},${aie_py_src},${aieargs}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp,-DCHANNELS=${channels} -DSEQ_LEN=${seq_len} -DKERNEL_SIZE=${kernel_size} -DBIAS=${bias}))

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE

.PHONY: clean
clean:
	rm -rf build _build ${targetname}.exe
