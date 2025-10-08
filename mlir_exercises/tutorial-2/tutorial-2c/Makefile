srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../makefile-common

.PHONY: all clean

all: tutorial-2c.exe
	@mkdir -p elf
	@mv *.elf* ./elf
	@cp ./elf/*.elf ./elf/*.elf.map .

# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
tutorial-2c.exe : ${srcdir}/test.cpp ${srcdir}/aie.mlir
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

Work/ps/c_rts/systemC/generated-objects/ps_i96.so: ${srcdir}/test.cpp ${srcdir}/aie.mlir
	make -C Work/ps/c_rts/systemC link

sim: Work/ps/c_rts/systemC/generated-objects/ps_i96.so
	aiesimulator --pkg-dir=./Work --dump-vcd foo


clean:
	rm -rf aie.mlir.prj aie_empty.mlir.prj aie_real_empty.mlir.prj *elf core* *log pl_sample_counts foo.vcd *exe aiesimulator_output sim .AIE_SIM_CMD_LINE_OPTIONS

#------------------------------------------------------------------------------
# Additional make targets for tutorial exercises
#------------------------------------------------------------------------------
tutorial-2c_perf.exe: ./answers/test_perf.cpp aie.mlir
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

tutorial-2c_empty_perf.exe: ./answers/test_perf.cpp ./answers/aie_empty.mlir
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

tutorial-2c_real_empty_perf.exe: ./answers/test_perf.cpp ./answers/aie_real_empty.mlir
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

