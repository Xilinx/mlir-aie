srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../makefile-common

devicename ?= npu2
targetname = pl_layout_test
in1_size = 256
out_size = 256
LAYOUT_VARIANT ?= 1

data_size = in1_size
aie_py_src = pl_test.py

.PHONY: all clean run

all: build/final_${data_size}.xclbin

build/pl_lut_kernel.o: ${srcdir}/pl_lut_kernel.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DLAYOUT_VARIANT=${LAYOUT_VARIANT} -DSTEP_BITS=${STEP_BITS} -c $< -o $(notdir $@)

build/aie2_pl_${data_size}.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< --in_size ${in1_size} > $@

build/final_${data_size}.xclbin: build/aie2_pl_${data_size}.mlir build/pl_lut_kernel.o
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=$(notdir $@) --npu-insts-name=insts_${data_size}.bin $(<:%=../%)

${targetname}_${data_size}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && cmake `dirname ${srcdir}/test.cpp` -DTARGET_NAME=${targetname} -DIN1_SIZE=${in1_size} -DOUT_SIZE=${out_size}
	cd _build && cmake --build . --config Release
	cp _build/${targetname} $@

run: ${targetname}_${data_size}.exe build/final_${data_size}.xclbin build/insts_${data_size}.bin
	./$< -x build/final_${data_size}.xclbin -i build/insts_${data_size}.bin -k MLIR_AIE

clean:
	rm -rf build _build ${targetname}*.exe
