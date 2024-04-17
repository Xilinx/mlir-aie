include ../../../makefile-common

mlirFileName = aie

all: build/conv2dk1_i8.o build/conv2dk1_skip_init.o build/conv2dk3.o build/conv2dk1_skip.o build/conv2dk1_ui8.o build/final.xclbin

# build/${mlirFileName}.mlir: aie2.py
# 	mkdir -p ${@D}
# 	python3 $< > $@

build/${mlirFileName}.mlir: aie.mlir
	mkdir -p ${@D}
	cp $< $@
insts.txt: build/${mlirFileName}.mlir
	aiecc.py -v --aie-only-generate-ipu --ipu-insts-name=$@ $<

build/conv2dk1_i8.o: conv2dk1.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DINT8_ACT -c $< -o $@

build/conv2dk3.o: conv2dk3.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DUINT8_ACT -c $< -o $@

build/conv2dk1_skip_init.o: conv2dk1_skip_init.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DINT8_ACT -c $< -o $@

build/conv2dk1_ui8.o: conv2dk1.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DUINT8_ACT -c $< -o $@

build/conv2dk1_skip.o: conv2dk1_skip.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DUINT8_ACT -c $< -o $@

build/final.xclbin: build/${mlirFileName}.mlir 
	cd build && aiecc.py -v --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

clean:
	rm -rf build *.elf* *.lst *.bif log* ${mlirFileName}.mlir.prj *.xclbin sim \
		chess* *.o insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe

run_py: 
	${powershell} python3 test.py
