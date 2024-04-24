include ../makefile-common

VPATH := $(VISION_KERNELS_VPATH_BASE)/threshold

all: final.xclbin

insts.txt: aie.mlir
	aiecc.py -v --aie-only-generate-npu --npu-insts-name=$@ $<

threshold.o: threshold.cc
	xchesscc -d ${CHESSCC2_FLAGS} -DBIT_WIDTH=8 -c $< -o $@
	
final.xclbin: aie.mlir threshold.o
	aiecc.py -v --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--xclbin-name=$@ --npu-insts-name=insts.txt $<

clean:
	rm -rf *.elf* *.bif aie.mlir.prj *.xclbin sim \
		chess* *.o* insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe

test.exe: test.cpp
	$(CXX) $(LD_FLAGS) $< -o $@ $(CFLAGS) $(INC) $(LD_PATHS) 
