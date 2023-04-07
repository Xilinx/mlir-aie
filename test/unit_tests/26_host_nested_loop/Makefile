include ../../../tutorials/makefile-common

.PHONY: all clean

all: host_nested_loop_SB.exe host_nested_loop_DB.exe

host_nested_loop_SB.exe: testSingleOutBuffer.cpp aieSingleOutBuffer.mlir 
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

host_nested_loop_DB.exe: testDoubleOutBuffer.cpp aieDoubleOutBuffer.mlir 
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

clean:
	rm -rf acdc_project *elf *exe core* sim .AIE_SIM_CMD_LINE_OPTIONS pl_sample_counts
