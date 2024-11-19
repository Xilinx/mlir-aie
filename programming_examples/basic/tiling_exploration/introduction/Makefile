#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: all clean

all: run

run: ${srcdir}/tilerhelper.ipynb
	mkdir -p build
	cd build && jupyter nbconvert --to script ${srcdir}/tilerhelper.ipynb --output-dir .
	cd build && ipython tilerhelper.py

clean:
	rm -rf build