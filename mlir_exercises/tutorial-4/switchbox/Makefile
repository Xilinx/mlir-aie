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

pathfinder:
	aie-opt --aie-canonicalize-device ./path/pathfinder_input.mlir | aie-opt --aie-create-pathfinder-flows > ./path/pathfinder_output.mlir
	aie-opt --aie-find-flows ./path/pathfinder_output.mlir | aie-translate --aie-flows-to-json > ./path/pathfinder_output.json

clean:
	rm -rf aie.mlir.prj aiesimulator_output *elf core* *log *vcd *exe pl_sample_counts .AIE_SIM_CMD_LINE_OPTIONS ./path/pathfinder_output.*
