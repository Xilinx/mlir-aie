srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../makefile-common

.PHONY: all clean

all: core_1_4.elf
	@mkdir -p elf
	@mv *.elf* ./elf
	@cp ./elf/*.elf ./elf/*.elf.map .

# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
core_1_4.elf : ${srcdir}/aie.mlir
	aiecc.py -j4 $<

clean:
	rm -rf aie.mlir.prj aiesimulator_output *elf core* *log *vcd pl_sample_counts .AIE_SIM_CMD_LINE_OPTIONS tutorial-1.exe

#------------------------------------------------------------------------------
# Additional make targets for tutorial exercises
# Note: AIECC_HOST_FLAGS is defined in the included tutorials/makefile-common.
#------------------------------------------------------------------------------
tutorial-1.exe: ${srcdir}/test.cpp ${srcdir}/aie.mlir
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

