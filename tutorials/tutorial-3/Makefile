
.PHONY: all clean

# TODO - Change this to env variable?
ifeq ($(MLIR_AIE_SYSROOT),)
MLIR_AIE_SYSROOT = /group/xrlabs/platforms/vck190-pynq-v2.7/sysroot
endif
AIE_INSTALL = $(dir $(shell which aie-opt))/..

all: tutorial-3.exe
	@mkdir -p elf
	@mv *.elf* ./elf
	@cp ./elf/*.elf ./elf/*.elf.map .

# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
tutorial-3.exe: test.cpp aie.mlir
	aiecc.py -j4 --sysroot=${MLIR_AIE_SYSROOT} --host-target=aarch64-linux-gnu $(word 2,$^) -I$(AIE_INSTALL)/runtime_lib $(AIE_INSTALL)/runtime_lib/test_library.cpp ./$< -o $@

clean:
	rm -rf acdc_project sim aiesimulator_output *elf core* *log *vcd *exe pl_sample_counts .AIE_SIM_CMD_LINE_OPTIONS

#------------------------------------------------------------------------------
# Additional make targets for tutorial exercises
#------------------------------------------------------------------------------
tutorial-3_perf1.exe: ./answers/test_perf.cpp ./aie.mlir
	aiecc.py -j4 --sysroot=${MLIR_AIE_SYSROOT} --host-target=aarch64-linux-gnu $(word 2,$^) -I$(AIE_INSTALL)/runtime_lib $(AIE_INSTALL)/runtime_lib/test_library.cpp ./$< -o $@

tutorial-3_perf2.exe: ./answers/test_perf.cpp ./answers/aie_perf.mlir
	aiecc.py -j4 --sysroot=${MLIR_AIE_SYSROOT} --host-target=aarch64-linux-gnu $(word 2,$^) -I$(AIE_INSTALL)/runtime_lib $(AIE_INSTALL)/runtime_lib/test_library.cpp ./$< -o $@
