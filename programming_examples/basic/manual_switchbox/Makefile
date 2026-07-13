##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Manual stream routing: every switchbox / shim_mux hop pinned by hand via a
# user-side Resolvable (no dedicated IRON API). The design's run_and_verify
# path compiles, runs, and checks against a numpy reference in one call, so the
# `run` target just invokes the script.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

SHELL := /bin/bash

targetname = manual_switchbox
devicename ?= npu2
COL ?= 0

aie_py_src = ${targetname}.py

.PHONY: all
all: run

.PHONY: run
run:
	python3 ${srcdir}/${aie_py_src} -d ${devicename} -c ${COL}

.PHONY: clean
clean:
	rm -rf build _build inst aie.mlir.prj core_* ${targetname}.exe
