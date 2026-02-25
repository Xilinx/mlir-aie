srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../makefile-common

.PHONY: all clean

all: tutorial-9.exe
	@mkdir -p elf
	@mv *.elf* ./elf
	@cp ./elf/*.elf ./elf/*.elf.map .

external_kernel/kernel.o: ${srcdir}/external_kernel/kernel.cc
	mkdir -p external_kernel
	cd external_kernel && $(MAKE) -f ${srcdir}/external_kernel/Makefile mlir

matmul_kernel/kernel.o: ${srcdir}/matmul_kernel/kernel.cc
	mkdir -p matmul_kernel
	cd matmul_kernel && $(MAKE) -f ${srcdir}/matmul_kernel/Makefile mlir

kernel.o: external_kernel/kernel.o
	cp external_kernel/kernel.o ./kernel.o

kernel_matmul.o: matmul_kernel/kernel.o
	cp matmul_kernel/kernel.o ./kernel_matmul.o

# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
tutorial-9.exe: ${srcdir}/test.cpp ${srcdir}/aie.mlir kernel.o
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

sim : ${srcdir}/test.cpp ${srcdir}/aie.mlir kernel.o
	aiecc.py -j4 --xchesscc --xbridge --aiesim $(word 2,$^) -I$(AIE_RUNTIME_LIB)/x86_64/test_lib/include -L$(AIE_RUNTIME_LIB)/x86_64/test_lib/lib -ltest_lib ./$<

clean:
	rm -rf external_kernel
	rm -rf matmul_kernel
	rm -rf aie.mlir.prj aiesimulator_output *elf core* *log *vcd *exe pl_sample_counts *.o .AIE_SIM_CMD_LINE_OPTIONS

#------------------------------------------------------------------------------
# Additional make targets for tutorial exercises
#------------------------------------------------------------------------------
tutorial-9_perf.exe: ${srcdir}/answers/test_perf.cpp ${srcdir}/aie.mlir kernel.o
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

tutorial-9_matmul_perf.exe: ${srcdir}/answers/test_matmul_perf.cpp ${srcdir}/answers/aie_matmul.mlir kernel_matmul.o
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@
