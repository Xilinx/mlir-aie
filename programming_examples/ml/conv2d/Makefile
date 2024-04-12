include ../../basic/makefile-common

#mlirFileName = aie_1core
mlirFileName = aieWithTrace_1core
# VPATH := ../../kernels

all: build/conv2dk1.o build/final.xclbin


build/${mlirFileName}.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< > $@

#insts.txt: ${mlirFileName}.mlir
insts.txt: build/${mlirFileName}.mlir
	aiecc.py -v --aie-only-generate-ipu --ipu-insts-name=$@ $<

build/conv2dk1.o: conv2dk1.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DINT8_ACT -c $< -o $@

build/final.xclbin: build/${mlirFileName}.mlir 
	cd build && aiecc.py -v --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

clean:
	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj *.xclbin sim \
		chess* *.o insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe
run_py: 
	${powershell} python3 test.py