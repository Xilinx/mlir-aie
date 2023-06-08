include ../makefile-common

.PHONY: all clean
 

all: tutorial-5.exe
	@mkdir -p elf
	@mv *.elf* ./elf
	@cp ./elf/*.elf ./elf/*.elf.map .

# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
tutorial-5.exe: aie.mlir test.cpp 
	aiecc.py -j4 $(AIECC_FLAGS) ./aie.mlir $(AIECC_HOST_FLAGS) ./test.cpp -o $@

sim : test.cpp aie.mlir
	aiecc.py -j4 --xchesscc --xbridge --aiesim $(word 2,$^) -I$(AIE_RUNTIME_LIB)/x86_64/test_lib/include -L$(AIE_RUNTIME_LIB)/x86_64/test_lib/lib -ltest_lib ./$<

clean:
	rm -rf aie.mlir.prj aiesimulator_output *elf core* *log *vcd *exe pl_sample_counts .AIE_SIM_CMD_LINE_OPTIONS
