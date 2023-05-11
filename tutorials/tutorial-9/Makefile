include ../makefile-common

.PHONY: all clean

all: tutorial-9.exe
	@mkdir -p elf
	@mv *.elf* ./elf
	@cp ./elf/*.elf ./elf/*.elf.map .

external_kernel/kernel.o: external_kernel/kernel.cc
	$(MAKE) -C external_kernel mlir

matmul_kernel/kernel.o: matmul_kernel/kernel.cc
	$(MAKE) -C matmul_kernel mlir

kernel.o: external_kernel/kernel.o
	cp external_kernel/kernel.o ./kernel.o

kernel_matmul.o: matmul_kernel/kernel.o
	cp matmul_kernel/kernel.o ./kernel_matmul.o

# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
tutorial-9.exe: test.cpp aie.mlir kernel.o
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

sim : test.cpp aie.mlir kernel.o
	aiecc.py -j4 --xchesscc --xbridge --aiesim $(word 2,$^) -I$(AIE_RUNTIME_LIB)/x86_64/test_lib/include -L$(AIE_RUNTIME_LIB)/x86_64/test_lib/lib -ltest_lib ./$<

clean:
	$(MAKE) -C external_kernel clean
	$(MAKE) -C matmul_kernel clean
	rm -rf aie.mlir.prj aiesimulator_output *elf core* *log *vcd *exe pl_sample_counts *.o .AIE_SIM_CMD_LINE_OPTIONS

#------------------------------------------------------------------------------
# Additional make targets for tutorial exercises
#------------------------------------------------------------------------------
tutorial-9_perf.exe: ./answers/test_perf.cpp ./aie.mlir kernel.o
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

tutorial-9_matmul_perf.exe: ./answers/test_matmul_perf.cpp ./answers/aie_matmul.mlir kernel_matmul.o
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@
