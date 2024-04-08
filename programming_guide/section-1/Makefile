##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../tutorials/makefile-common

build/aie.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< > $@ 

all: build/aie.mlir

clean:
	rm -rf build 
