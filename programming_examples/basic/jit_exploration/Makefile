#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: all clean

all: run

run: ${srcdir}/taplib.ipynb
	mkdir -p build
	cd build && jupyter nbconvert --to script ${srcdir}/taplib.ipynb --output-dir .
	cd build && ipython taplib.py

clean:
	rm -rf build