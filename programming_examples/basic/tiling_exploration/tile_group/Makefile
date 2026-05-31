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
# Tile-group tensor access exploration.  @iron.jit drives aiecc directly
# via --xclbin-path / --insts-path; standalone `python3 tile_group.py`
# also JITs + verifies.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../../makefile-common

SHELL := /bin/bash

tensor_height ?= 8
tensor_width ?= 8
tile_height ?= 2
tile_width ?= 2

data_str = ${tensor_height}_${tensor_width}_${tile_height}_${tile_width}
aie_py_src = tile_group.py
aieargs = -d ${devicename} --tensor-height ${tensor_height} --tensor-width ${tensor_width} --tile-height ${tile_height} --tile-width ${tile_width}

xclbin_target = build/final_${data_str}.xclbin
insts_target  = build/insts_${data_str}.bin

.PHONY: all
all: ${xclbin_target} ${insts_target}

$(eval $(call jit_xclbin,${xclbin_target},${insts_target},${aie_py_src},${aieargs}))

.PHONY: run_py
run_py: ${srcdir}/${aie_py_src}
	${powershell} python3 $< ${aieargs}

.PHONY: generate_access_map
generate_access_map: ${srcdir}/${aie_py_src}
	python3 $< ${aieargs} --generate-access-map

.PHONY: clean
clean:
	rm -rf build
