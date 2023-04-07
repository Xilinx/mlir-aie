include ../../../tutorials/makefile-common

.PHONY: all clean

all: host_multirate_SB.exe host_multirate_DB.exe
host_multirate_SB.exe: testSingleOutBuffer.cpp aieSingleOutBuffer.mlir 
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

host_multirate_DB.exe: testDoubleOutBuffer.cpp aieDoubleOutBuffer.mlir 
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

clean:
	rm -rf acdc_project *elf *exe core* sim *.log .AIE_SIM_CMD_LINE_OPTIONS pl_sample_counts
