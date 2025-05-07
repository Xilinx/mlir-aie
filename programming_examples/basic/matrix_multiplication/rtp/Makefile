##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

# ---

# The following environment variables that point to the Xilinx runtime (XRT)
# should be set up by an environment setup script already.
XILINX_XRT?=/opt/xilinx/xrt
XILINX_VITIS?=/tools/Xilinx/Vitis/2023.2
BOOST_ROOT?=/usr/include/boost
Boost_INCLUDE_DIRS=${BOOST_ROOT}

# ---

XILINX_XRT_INCLUDE?=${XILINX_XRT}/include
XILINX_XRT_LIB?=${XILINX_XRT}/lib

CXX=g++-13

CXXFLAGS+=-std=gnu++23 -ggdb -I${XILINX_XRT_INCLUDE} ${Boost_INCLUDE_DIRS:%=-I%}
LDFLAGS+=-L${XILINX_XRT_LIB} ${Boost_LIBRARY_DIRS:%=-L%}
LDLIBS+=-lxrt_coreutil -lboost_program_options -lboost_filesystem -luuid

CHESSCCWRAP2_FLAGS = aie2 -I${XILINX_VITIS}/aietools/include 

mlir_target?=build/aie.mlir
xclbin_target?=build/final.xclbin
insts_target?=build/insts.txt
kernel_target?=build/mm_${m}x${k}x${n}.o

M=256
K=3072
N=50304
m=64
k=64
n=32
n_aie_cols=4
dtype_in=bf16
dtype_out=f32

aieargs+=-m $m -k $k -n $n --n-aie-cols ${n_aie_cols} --dtype_in ${dtype_in} --dtype_out ${dtype_out}

.PHONY: all
all: test ${xclbin_target} ${host_target} \
	build/insts_256x768x2304.txt \
	build/insts_256x768x768.txt \
	build/insts_256x768x3072.txt \
	build/insts_256x3072x768.txt \
	build/insts_768x256x3072.txt \
	build/insts_3072x256x768.txt \
	build/insts_768x256x768.txt \
	build/insts_256x2304x768.txt \
	build/insts_2304x256x768.txt

${kernel_target}: kernel.cc await_rtp.cc zero.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DDIM_M=$m -DDIM_K=$k -DDIM_N=$n -DBIT_WIDTH=8 -c ${<:%=../%} -o ${@F}

${xclbin_target}: build/aie_256x768x2304.mlir ${kernel_target} 
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-cdo --no-compile-host --xclbin-name=${@F}  $(<:%=../%)

build/aie_256x768x2304.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< -M 256 -K 768 -N 2304 $(aieargs) > $@
build/aie_256x768x768.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< -M 256 -K 768 -N 768 $(aieargs) > $@
build/aie_256x768x3072.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< -M 256 -K 768 -N 3072 $(aieargs) > $@
build/aie_256x3072x768.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< -M 256 -K 3072 -N 768 $(aieargs) > $@
build/aie_768x256x3072.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< -M 768 -K 256 -N 3072 $(aieargs) > $@
build/aie_3072x256x768.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< -M 3072 -K 256 -N 768 $(aieargs) > $@
build/aie_768x256x768.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< -M 768 -K 256 -N 768 $(aieargs) > $@
build/aie_256x2304x768.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< -M 256 -K 2304 -N 768 $(aieargs) > $@
build/aie_2304x256x768.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< -M 2304 -K 256 -N 768 $(aieargs) > $@

build/insts_%.txt: build/aie_%.mlir 
	cd ${@D} && aiecc.py -v --aie-only-generate-npu --npu-insts-name=${@:build/%=%} $(<:%=../%)

xclbin_sign=${XILINX_XRT}/amdxdna/setup_xclbin_firmware.sh 
.PHONY: sign
sign: ${xclbin_target}
	${xclbin_sign} -dev Phoenix -xclbin $<

.PHONY: clean
clean:
	@rm -r build

test: test.cpp 
	${CXX} ${CXXFLAGS} $^ -o $@ ${LDFLAGS} ${LDLIBS}

.PHONY: run
run: all
	./test
