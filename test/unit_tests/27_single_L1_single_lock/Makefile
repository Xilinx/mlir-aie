include ../../../tutorials/makefile-common

.PHONY: all clean

all: host_single1SingleLock.exe host_single1SingleLock_FIX.exe

host_single1SingleLock.exe: test.cpp aie.mlir 
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

host_single1SingleLock_FIX.exe: test.cpp aieWithWorkaround.mlir 
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@


clean:
	rm -rf acdc_project *elf *exe core* sim .AIE_SIM_CMD_LINE_OPTIONS pl_sample_counts *.log *.vcd
