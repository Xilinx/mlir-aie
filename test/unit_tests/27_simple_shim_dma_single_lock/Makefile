include ../../../tutorials/makefile-common

.PHONY: all clean

all: host_simpleShimSingleLock.exe

host_simpleShimSingleLock.exe: test.cpp aie.mlir 
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@


clean:
	rm -rf acdc_project *elf *exe core* sim .AIE_SIM_CMD_LINE_OPTIONS pl_sample_counts
