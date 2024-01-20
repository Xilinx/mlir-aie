include ../../../makefile-common

# mlirFileName = aieDemo2x
mlirFileName = aieFinalx2
# mlirFileName = aieFinalx2_inverted

# mlirFileName = aieFinalx1+conv1
# mlirFileName = aieFinalx1+conv1_conv3

all: build/conv2dk1.o build/conv2dk1_skip_init.o build/conv2dk3.o build/conv2dk1_skip.o build/final.xclbin
VPATH := ../../kernels

insts.txt: ${mlirFileName}.mlir
	aiecc.py -v --aie-only-generate-ipu --ipu-insts-name=$@ $<

build/conv2dk1.o: conv2dk1.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -INT8_ACT -c $< -o $@

build/conv2dk3.o: conv2dk3.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -UINT8_ACT -c $< -o $@

build/conv2dk1_skip_init.o: conv2dk1_skip_init.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -INT8_ACT -c $< -o $@

build/conv2dk1.o: conv2dk1.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -UINT8_ACT -c $< -o $@

build/conv2dk1_skip.o: conv2dk1_skip.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DSCALAR -UINT8_ACT -c $< -o $@

build/final.xclbin: ${mlirFileName}.mlir 
	cd build && aiecc.py -v --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

clean:
	rm -rf build *.elf* *.lst *.bif *.mlir.prj *.xclbin sim \
		chess* *.o insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe

build/test.exe: test.cpp
	$(CXX) $(LD_FLAGS) $< -o $@ $(CFLAGS) $(INC) $(LD_PATHS) 

