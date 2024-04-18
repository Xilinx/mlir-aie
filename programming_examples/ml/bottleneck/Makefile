include ../../makefile-common

mlirFileName = aie

all: build/conv2dk1.o build/conv2dk3.o build/conv2dk1_skip.o build/final.xclbin

build/${mlirFileName}.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< > $@

insts.txt: build/${mlirFileName}.mlir
	aiecc.py -v --aie-only-generate-ipu --ipu-insts-name=$@ $<

build/conv2dk1.o: ../../../aie_kernels/aie2/conv2dk1.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DINT8_ACT -c $< -o $@

build/conv2dk3.o: ../../../aie_kernels/aie2/conv2dk3.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DUINT8_ACT -c $< -o $@

build/conv2dk1_skip.o: ../../../aie_kernels/aie2/conv2dk1_skip.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DINT8_ACT -c $< -o $@

build/final.xclbin: build/${mlirFileName}.mlir 
	cd build && aiecc.py -v --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

clean:
	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj log .xclbin sim \
		chess* *.o insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe

run_py: 
	${powershell} python3 test.py
