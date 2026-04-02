##===- Makefile -----------------------------------------------------------===##
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../../makefile-common
VPATH := ${srcdir}/../../../../aie_kernels/generic
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

.PHONY: all clean run_py

build/passThrough.cc.o: passThrough.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
endif

build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< -d ${devicename} > $@

build/final.xclbin: build/aie.mlir build/passThrough.cc.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=${@F} --npu-insts-name=insts.bin $(<:%=../%)

all: build/final.xclbin

run_py: build/final.xclbin
	python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE

clean:
	rm -rf build _build
