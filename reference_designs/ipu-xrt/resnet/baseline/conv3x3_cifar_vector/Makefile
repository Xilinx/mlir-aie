include ../../../makefile-common

#mlirFileName = bottlenck4AIEs
# mlirFileName = aieWithTrace
mlirFileName = aie
VPATH := ../../kernels

all: build/pass_through.o build/conv2dk3.o build/final.xclbin

insts.txt: ${mlirFileName}.mlir
	aiecc.py -v --aie-only-generate-ipu --ipu-insts-name=$@ $<

build/pass_through.o: pass_through.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DINT8_ACT -c $< -o $@
build/conv2dk3.o: conv2dk3.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DINT8_ACT -c $< -o $@

build/final.xclbin: ${mlirFileName}.mlir 
	cd build && aiecc.py -v --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=${@F} --ipu-insts-name=insts.txt $(<:%=../%)

clean:
	rm -rf build *.elf* *.lst *.bif *.mlir.prj *.xclbin sim \
		chess* *.o insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe

build/test.exe: test.cpp
	$(CXX) $(LD_FLAGS) $< -o $@ $(CFLAGS) $(INC) $(LD_PATHS) 

