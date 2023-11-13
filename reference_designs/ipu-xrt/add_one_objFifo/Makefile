include ../makefile-common

all: final.xclbin

final.xclbin: aie.mlir
	aiecc.py --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=$@ --ipu-insts-name=insts.txt $<

clean:
	rm -rf *.elf* *.bif aie.mlir.prj *.xclbin insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe

test.exe: test.cpp
	$(CXX) $(LD_FLAGS) $< -o $@ $(CFLAGS) $(INC) $(LD_PATHS) 
