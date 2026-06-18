##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##
#
# Four-way transpose demo (dma, dma_packet, shuffle, combined).  Pick
# one via STRATEGY=...; per-strategy defaults for M / K / dtype-bytes
# live in transposes.py's _DEFAULTS so `make run` Just Works for any
# strategy.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

targetname = transposes

STRATEGY ?= dma

# Per-strategy defaults — keep in sync with transposes.py's _DEFAULTS.
# Caller can still override on the make cmdline (e.g. M=128 K=64).
ifeq (${STRATEGY},dma)
  M ?= 64
  K ?= 64
  DTYPE_BYTES ?= 4
endif
ifeq (${STRATEGY},dma_packet)
  M ?= 64
  K ?= 32
  DTYPE_BYTES ?= 4
endif
ifeq (${STRATEGY},shuffle)
  M ?= 16
  K ?= 16
  DTYPE_BYTES ?= 1
endif
ifeq (${STRATEGY},combined)
  M ?= 128
  K ?= 128
  DTYPE_BYTES ?= 4
  COMBINED_ARGS ?= -m 32 -n 32 --ss 8
endif

aie_py_src = ${targetname}.py
aieargs = -d ${devicename} -s ${STRATEGY} -M ${M} -K ${K} --dtype-bytes ${DTYPE_BYTES} ${COMBINED_ARGS}

xclbin_target = build/${STRATEGY}/final.xclbin
insts_target  = build/${STRATEGY}/insts.bin

.PHONY: all
all: ${xclbin_target} ${insts_target}

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))
$(eval $(call build_host_exe,${targetname},_build,test.cpp))

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -M ${M} -K ${K} --dtype-bytes ${DTYPE_BYTES}

.PHONY: clean
clean:
	rm -rf build _build inst ${targetname}.exe
