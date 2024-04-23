##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

include ../../makefile-common

VPATH := ../../../aie_kernels/aie2

targetname = vectorScalar
#data_size = 4096
data_size = 512
#data_size = 1024
trace_size = 8192

all: build/final_${data_size}.xclbin build/insts_${data_size}.txt

kristof: build/insts_${data_size}.txt

build/%.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $(<:%=../%) -o ${@F}

build/aie_${data_size}.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< ${data_size} 0 > $@

build/aie_trace_${data_size}.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< ${data_size} ${trace_size} > $@

#build/insts_${data_size}.txt: build/final_${data_size}.xclbin
build/final_${data_size}.xclbin: build/aie_${data_size}.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=insts_${data_size}.txt $(<:%=../%)

build/final_trace_${data_size}.xclbin: build/aie_trace.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=insts_${data_size}.txt $(<:%=../%)

${targetname}_${data_size}.exe: test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake .. -DTARGET_NAME=${targetname}_${data_size} -DVECTORSCALARMUL_SIZE=${data_size}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}_${data_size}.exe $@
else
	cp _build/${targetname}_${data_size} $@ 
endif

run: ${targetname}_${data_size}.exe build/final_${data_size}.xclbin build/insts_${data_size}.txt 
	${powershell} ./$< -x build/final_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE

run_py: build/final_${data_size}.xclbin build/insts_${data_size}.txt
	${powershell} python3 test.py -x build/final_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE -s ${data_size}

trace: ${targetname}_${data_size}.exe build/final_trace_${data_size}.xclbin build/insts_${data_size}.txt 
	${powershell} ./$< -x build/final_trace_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE -t ${trace_size}
	../../utils/parse_eventIR.py --filename trace.txt --mlir build/aie_trace__${data_size}.mlir --colshift 1 > parse_eventIR_vs.json

trace_py: build/final_trace_${data_size}.xclbin build/insts_${data_size}.txt
	${powershell} python3 test.py -x build/final_trace_${data_size}.xclbin -i build/insts_${data_size}.txt -k MLIR_AIE -t ${trace_size} -s ${data_size}
	../../utils/parse_eventIR.py --filename trace.txt --mlir build/aie_trace_${data_size}.mlir --colshift 1 > parse_eventIR_vs.json


clean_trace:
	rm -rf tmpTrace trace.txt

clean: clean_trace
	rm -rf build _build ${targetname}_*.exe
