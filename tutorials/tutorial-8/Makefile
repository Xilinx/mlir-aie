include ../makefile-common

.PHONY: all clean

ifeq (${AIETOOLS_ROOT},)
AIETOOLS_ROOT = $(dir $(shell which xchesscc))/..
endif

all: tutorial-8.exe
	@mkdir -p elf
	@mv *.elf* ./elf
	@cp ./elf/*.elf ./elf/*.elf.map .

# Command line AIE kernel compile. See tutorial-2 for more info.
%.o: %.cc
	xchesscc -p me -P ${AIETOOLS_ROOT}/data/cervino/lib -c $<

# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
tutorial-8.exe: test.cpp aie.mlir kernel1.o kernel2.o
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

clean:
	rm -rf acdc_project sim aiesimulator_output *elf core* *log *vcd *exe pl_sample_counts *.o .AIE_SIM_CMD_LINE_OPTIONS

#------------------------------------------------------------------------------
# Additional make targets for tutorial exercises
#------------------------------------------------------------------------------
tutorial-8_q3.exe: test.cpp ./answers/aie.mlir
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@
