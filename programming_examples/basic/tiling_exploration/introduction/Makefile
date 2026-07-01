# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: all clean

all: run

run: ${srcdir}/taplib.ipynb
	mkdir -p build
	cd build && jupyter nbconvert --to script ${srcdir}/taplib.ipynb --output-dir .
	cd build && ipython taplib.py

clean:
	rm -rf build