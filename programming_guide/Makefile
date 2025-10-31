#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: all clean

all: run

run: ${srcdir}/jit_tutorial.ipynb
	mkdir -p build
	cd build && jupyter nbconvert --to script ${srcdir}/jit_tutorial.ipynb --output-dir .
	cd build && ipython jit_tutorial.py

clean:
	rm -rf build
