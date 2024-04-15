include ../../makefile-common

mlirFileName = aie

all: build/conv2dk1.o build/conv2dk3.o build/conv2dk1_skip.o build/final.xclbin

build/${mlirFileName}.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< > $@

# build/${mlirFileName}.mlir: aie.mlir
# 	mkdir -p ${@D}
# 	cp $< $@
#insts.txt: ${mlirFileName}.mlir
insts.txt: build/${mlirFileName}.mlir
	aiecc.py -v --aie-only-generate-ipu --ipu-insts-name=$@ $<

build/conv2dk1.o: conv2dk1.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DINT8_ACT -c $< -o $@

build/conv2dk3.o: conv2dk3.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DUINT8_ACT -c $< -o $@

build/conv2dk1_skip.o: conv2dk1_skip.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DINT8_ACT -c $< -o $@

build/final.xclbin: build/${mlirFileName}.mlir 
	cd build && aiecc.py -v --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

# clean:
# 	rm -rf build *.elf* *.lst *.bif ${mlirFileName}.mlir.prj *.xclbin sim \
# 		chess* *.o insts.txt \
# 		*.log aie_partition.json *.bin BOOT.BIN _x test.exe

clean:
	rm -rf  build/*.elf* build/*.lst build/*.bif build/${mlirFileName}.mlir.prj build/*.xclbin build/sim \
		build/chess* *.o build/insts.txt \
		build/*.log build/aie_partition.json build/*.bin build/BOOT.BIN build/_x test.exe
run_py: 
	${powershell} python3 test.py
