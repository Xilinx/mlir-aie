##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
#
# Build the AIE2P dispatch-overhead bisector example xclbin + host runner.
#
# Single-tile passthrough kernel + (N_CHUNKS, DENSE_BYTES)-parameterised
# IRON topology. The ``all-variants`` target rebuilds the xclbin once per
# (N_CHUNKS, DENSE_BYTES) pair so a driver script can dispatch each
# variant under the same host runner binary and regress to attribute the
# per-launch wall to its dispatch-floor suspects.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

devicename ?= npu2
targetname = dispatch_overhead_bisector

aie_py_src = dispatch_overhead_bisector.py
aie_cc_src = passthrough.cc
runner_cc_src = test.cpp

# Per-variant build directory: build/n${N_CHUNKS}_b${DENSE_BYTES}/{aie.mlir,
# final.xclbin, insts.bin}. Default n=1, b=4096.
N_CHUNKS ?= 1
DENSE_BYTES ?= 4096
variant_dir = build/n$(N_CHUNKS)_b$(DENSE_BYTES)

all: $(variant_dir)/final.xclbin $(variant_dir)/insts.bin ${targetname}

# all-variants: build a representative bisection sweep.
#
# shim-chunks sweep at fixed dense_bytes=4096 (probes suspect c
# per-chunk shim-DMA setup; AIE2P shim BD pool typically caps useful
# n_chunks at 4):
#   n=1, n=2, n=4
#
# dense-bytes sweep at fixed n=1 (probes payload-bytes scaling vs
# per-launch fixed cost):
#   b=512, b=4096, b=8192
all-variants:
	$(MAKE) -C $(srcdir) variant N_CHUNKS=1  DENSE_BYTES=4096
	$(MAKE) -C $(srcdir) variant N_CHUNKS=2  DENSE_BYTES=4096
	$(MAKE) -C $(srcdir) variant N_CHUNKS=4  DENSE_BYTES=4096
	$(MAKE) -C $(srcdir) variant N_CHUNKS=1  DENSE_BYTES=512
	$(MAKE) -C $(srcdir) variant N_CHUNKS=1  DENSE_BYTES=8192
	$(MAKE) -C $(srcdir) ${targetname}

variant: $(variant_dir)/final.xclbin $(variant_dir)/insts.bin

build/passthrough.cc.o: ${aie_cc_src}
	mkdir -p $(@D)
ifeq ($(devicename),npu2)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} \
	  -I${MLIR_AIE_DIR}/aie_kernels \
	  -I${MLIR_AIE_DIR}/install/include \
	  -c ${srcdir}/${aie_cc_src} -o passthrough.cc.o
else
	$(error devicename=$(devicename) not supported; AIE2P-only)
endif

$(variant_dir)/aie.mlir: ${aie_py_src}
	mkdir -p $(@D)
	python3 ${srcdir}/${aie_py_src} -d ${devicename} \
	  --dense-bytes $(DENSE_BYTES) --n-chunks $(N_CHUNKS) > $@

# aiecc emits the xclbin + insts.bin in the CWD it's invoked from.
# Invoke from $(variant_dir) so all artefacts stay co-located, and copy
# the per-variant passthrough.cc.o in (xclbin packaging picks it up by
# relative path from the MLIR).
$(variant_dir)/final.xclbin: $(variant_dir)/aie.mlir build/passthrough.cc.o
	mkdir -p $(@D)
	cp build/passthrough.cc.o $(@D)/passthrough.cc.o
	cd $(@D) && aiecc --aie-generate-xclbin --no-compile-host \
	  --xclbin-name=$(@F) \
	  --no-xchesscc --no-xbridge \
	  --aie-generate-npu-insts --npu-insts-name=insts.bin \
	  aie.mlir

$(variant_dir)/insts.bin: $(variant_dir)/final.xclbin
	@test -f $@ || (echo "expected $@ as a side effect of the xclbin build" && exit 1)

${targetname}: ${runner_cc_src}
	@if [ -z "$$XILINX_XRT" ]; then \
	  echo "XILINX_XRT not set; source /opt/xilinx/xrt/setup.sh first" >&2; \
	  exit 1; \
	fi
	${CXX} -std=c++17 -O2 \
	  -I${XILINX_XRT}/include \
	  ${srcdir}/${runner_cc_src} \
	  -L${XILINX_XRT}/lib -Wl,-rpath,${XILINX_XRT}/lib \
	  -lxrt_coreutil -luuid \
	  -o ${targetname}

clean:
	rm -rf build ${targetname}

.PHONY: all all-variants variant clean
