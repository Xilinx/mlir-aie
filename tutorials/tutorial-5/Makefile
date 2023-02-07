
.PHONY: all clean

# TODO - Change this to env variable?
ifeq ($(MLIR_AIE_SYSROOT),)
MLIR_AIE_SYSROOT = /group/xrlabs/platforms/vck190-pynq-v2.7/sysroot
endif
AIE_INSTALL = $(dir $(shell which aie-opt))/..

all: tutorial-5.exe

# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
tutorial-5.exe: aie.mlir test.cpp 
	aiecc.py -j4 --sysroot=${MLIR_AIE_SYSROOT} --host-target=aarch64-linux-gnu ./aie.mlir -I$(AIE_INSTALL)/runtime_lib $(AIE_INSTALL)/runtime_lib/test_library.cpp ./test.cpp -o $@

clean:
	rm -rf acdc_project sim aiesimulator_output *elf core* *log *vcd *exe pl_sample_counts
