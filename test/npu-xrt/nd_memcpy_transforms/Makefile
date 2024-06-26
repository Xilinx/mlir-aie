# ---

# The following environment variables that point to the Xilinx runtime (XRT)
# should be set up by an environment setup script already.
XILINX_XRT?=/opt/xilinx/xrt
XILINX_VITIS?=/tools/Xilinx/Vitis/2023.2

# ---

XILINX_XRT_INCLUDE?=${XILINX_XRT}/include
XILINX_XRT_LIB?=${XILINX_XRT}/lib

CXX=g++-13 -ggdb -I${XILINX_XRT_INCLUDE} -L${XILINX_XRT_LIB}

CHESSCCWRAP2_FLAGS = aie2 -I${XILINX_VITIS}/aietools/include 

mlir_target?=build/aie.mlir
xclbin_target?=build/final.xclbin
insts_target?=build/insts.txt
host_target?=build/test
kernels?=kernel.cc

.PHONY: all
all: ${xclbin_target} ${host_target}

build/%.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBIT_WIDTH=8 -c ${<:%=../%} -o ${@F}

${mlir_target}: aie2.py
	mkdir -p ${@D}
	python3 $< > $@

${xclbin_target}: ${mlir_target} ${kernels:%.cc=build/%.o}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=${insts_target:build/%=%} $(<:%=../%)

${host_target}: test.cpp
	mkdir -p ${@D}
	${CXX} -o $@ $< -lxrt_coreutil

xclbin_sign=${XILINX_XRT}/amdxdna/setup_xclbin_firmware.sh 
.PHONY: sign
sign: ${xclbin_target}
	${xclbin_sign} -dev Phoenix -xclbin $<

.PHONY: run
run: ${host_target} ${xclbin_target}
	cd $(dir ${host_target}) && ./$(notdir ${host_target})

.PHONY: clean
clean:
	rm -r build
