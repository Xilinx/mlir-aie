include ../../../tutorials/makefile-common

.PHONY: all clean

all: SingleLock.exe SingleLock_FIX.exe

SingleLock.exe: test.cpp aie.mlir
	aiecc.py -v -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

SingleLock_FIX.exe: test.cpp aieWithWorkaround.mlir
	aiecc.py -v -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

SingleLock_aie2.exe: test.cpp aie2.mlir
	aiecc.py -v -j4 --aie-target=AIE2 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@



clean:
	rm -rf aie.mlir.prj aieWithWorkaround.mlir.prj aie2.mlir.prj aiesimulator_output *elf *exe core* sim .AIE_SIM_CMD_LINE_OPTIONS pl_sample_counts *.log *.vcd
