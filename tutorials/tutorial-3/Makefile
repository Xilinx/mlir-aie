
.PHONY: all clean

# TODO - Change this to env variable?
SYSROOT = /group/xrlabs/platforms/vck190-pynq-v2.7/sysroot

all: tutorial-3.exe

# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
tutorial-3.exe: aie.mlir test.cpp
	aiecc.py -j4 --sysroot=${SYSROOT} --host-target=aarch64-linux-gnu ./aie.mlir -I../../runtime_lib ../../runtime_lib/test_library.cpp ./test.cpp -o $@

clean:
	rm -rf acdc_project *elf *exe core*

#------------------------------------------------------------------------------
# Additional make targets for tutorial exercises
#------------------------------------------------------------------------------
tutorial-3_perf.exe: ./answers/test_perf.cpp ./aie.mlir
	aiecc.py -j4 --sysroot=${SYSROOT} --host-target=aarch64-linux-gnu $(word 2,$^) -I../../runtime_lib ../../runtime_lib/test_library.cpp ./$< -o $@
