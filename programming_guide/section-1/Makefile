##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../programming_examples/makefile-common

build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< > $@ 

build/aie_placed.mlir: ${srcdir}/aie2_placed.py
	mkdir -p ${@D}
	python3 $< > $@ 

all: build/aie.mlir

placed: build/aie_placed.mlir

clean:
	rm -rf build 
