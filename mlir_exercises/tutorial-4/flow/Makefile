srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../makefile-common

.PHONY: all clean

all: tutorial-4.exe
	@mkdir -p elf
	@mv *.elf* ./elf
	@cp ./elf/*.elf ./elf/*.elf.map .

# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
tutorial-4.exe: ${srcdir}/test.cpp ${srcdir}/aie.mlir
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

clean:
	rm -rf aie.mlir.prj aie_q5.mlir.prj aiesimulator_output *elf core* *log *vcd *exe pl_sample_counts .AIE_SIM_CMD_LINE_OPTIONS

#------------------------------------------------------------------------------
# Additional make targets for tutorial exercises
#------------------------------------------------------------------------------
tutorial-4_q5.exe: answers/test_q5.cpp answers/aie_q5.mlir 
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

tutorial-4_perf.exe: answers/test_perf.cpp aie.mlir 
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

