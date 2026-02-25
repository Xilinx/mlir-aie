srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../makefile-common

.PHONY: all clean

all: tutorial-2a.exe
	@mkdir -p elf
	@mv *.elf* ./elf
	@cp ./elf/*.elf ./elf/*.elf.map .

# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
tutorial-2a.exe : ${srcdir}/test.cpp ${srcdir}/aie.mlir
	aiecc.py -j4 ${AIECC_FLAGS} $(word 2,$^) ${AIECC_HOST_FLAGS} ./$< -o $@

sim : ${srcdir}/test.cpp ${srcdir}/aie.mlir
	aiecc.py -j4 --xchesscc --xbridge --aiesim $(word 2,$^) -I$(AIE_RUNTIME_LIB)/x86_64/test_lib/include -L$(AIE_RUNTIME_LIB)/x86_64/test_lib/lib -ltest_lib ./$<

clean:
	rm -rf aie.mlir.prj *elf core* *log pl_sample_counts foo.vcd *exe aiesimulator_output sim .AIE_SIM_CMD_LINE_OPTIONS


