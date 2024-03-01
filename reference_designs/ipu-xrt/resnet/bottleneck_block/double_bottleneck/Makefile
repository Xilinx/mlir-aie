include ../../../makefile-common

# mlirFileName = aieDemo2x
# mlirFileName = aieFinalx2
mlirFileName = aieFinalx2Inverted

# mlirFileName = aieFinalx1+conv1
# mlirFileName = aieFinalx1+conv1_conv3

all: build/conv2dk1_i8.o build/conv2dk1_skip_init_i8.o build/conv2dk3_ui8.o build/conv2dk1_ui8.o build/conv2dk1_skip_ui8.o build/final.xclbin
VPATH := ../../kernels

insts.txt: ${mlirFileName}.mlir
	aiecc.py -v --aie-only-generate-ipu --ipu-insts-name=$@ $<

build/conv2dk1_i8.o: scalar_conv2dk1_i8.cc
	xchesscc -d ${CHESSCC2_FLAGS} -c $< -o $@

build/conv2dk3_ui8.o: scalar_conv2dk3_ui8.cc
	xchesscc -d ${CHESSCC2_FLAGS} -c $< -o $@

build/conv2dk1_skip_init_i8.o: scalar_conv2dk1_skip_init_i8.cc
	xchesscc -d ${CHESSCC2_FLAGS} -c $< -o $@

build/conv2dk1_ui8.o: scalar_conv2dk1_ui8.cc
	xchesscc -d ${CHESSCC2_FLAGS} -c $< -o $@

build/conv2dk1_skip_ui8.o: scalar_conv2dk1_skip_ui8.cc
	xchesscc -d ${CHESSCC2_FLAGS} -c $< -o $@

build/final.xclbin: ${mlirFileName}.mlir 
	cd build && aiecc.py -v --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

clean:
	rm -rf build *.elf* *.lst *.bif *.mlir.prj *.xclbin sim \
		chess* *.o insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe

build/test.exe: test.cpp
	$(CXX) $(LD_FLAGS) $< -o $@ $(CFLAGS) $(INC) $(LD_PATHS) 

