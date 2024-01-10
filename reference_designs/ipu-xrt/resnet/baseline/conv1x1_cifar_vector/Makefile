include ../../../makefile-common

mlirFileName = aie_1core
all: conv2dk1.o final.xclbin

insts.txt: ${mlirFileName}.mlir
	aiecc.py -v --aie-only-generate-ipu --ipu-insts-name=$@ $<

conv2dk1.o: conv2dk1.cc
	xchesscc -d ${CHESSCC2_FLAGS} -c $< -o $@

final.xclbin: ${mlirFileName}.mlir 
	aiecc.py -v --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=$@ --ipu-insts-name=insts.txt $<

clean:
	rm -rf *.elf* *.lst *.bif ${mlirFileName}.mlir.prj *.xclbin sim \
		chess* *.o insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe

test.exe: test.cpp
	$(CXX) $(LD_FLAGS) $< -o $@ $(CFLAGS) $(INC) $(LD_PATHS) 

# insts.txt: generated_ipu_insts.mlir
# 	aie-translate --aie-ipu-instgen -o $@ $<
