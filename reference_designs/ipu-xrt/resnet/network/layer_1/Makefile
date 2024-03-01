include ../../../makefile-common

# mlirFileName = aieFinal
# mlirFileName = aieDemo
mlirFileName = aieResnet
# mlirFileName = aieFinalWithTrace

all: build/conv2dk1_i8.o build/conv2dk1_skip_init.o build/conv2dk3.o build/conv2dk1_skip.o build/conv2dk1_ui8.o build/final.xclbin
VPATH := ../../kernels

insts.txt: ${mlirFileName}.mlir
	aiecc.py -v --aie-only-generate-ipu --ipu-insts-name=$@ $<

build/conv2dk1_i8.o: conv2dk1.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DINT8_ACT -c $< -o $@

build/conv2dk3.o: conv2dk3.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DUINT8_ACT -c $< -o $@

build/conv2dk1_skip_init.o: conv2dk1_skip_init.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DINT8_ACT -c $< -o $@

build/conv2dk1_ui8.o: conv2dk1.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DUINT8_ACT -c $< -o $@

build/conv2dk1_skip.o: conv2dk1_skip.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -DUINT8_ACT -c $< -o $@

build/final.xclbin: ${mlirFileName}.mlir 
	cd build && aiecc.py -v --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

clean:
	rm -rf build/*.elf* build/*.lst build/*.bif build/*.mlir.prj build/*.xclbin build/sim \
		build/chess* build/insts.txt \
		build/*.log build/aie_partition.json build/*.bin build/BOOT.BIN _x build/test.exe
# clean:
# 	rm -rf build *.elf* *.lst *.bif *.mlir.prj *.xclbin sim \
# 		chess* *.o insts.txt \
# 		*.log aie_partition.json *.bin BOOT.BIN _x test.exe

build/test.exe: test.cpp
	$(CXX) $(LD_FLAGS) $< -o $@ $(CFLAGS) $(INC) $(LD_PATHS) 

