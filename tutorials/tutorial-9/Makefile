
.PHONY: all clean

# TODO - Change this to env variable?
SYSROOT = /group/xrlabs/platforms/vck190-pynq-v2.7/sysroot
AIE_INSTALL = $(dir $(shell which aie-opt))/..

all: tutorial-9.exe

#external_kernel/work/Release_LLVM/test.prx/kernel.o: external_kernel/kernel.cc
#	$(MAKE) -C external_kernel build

external_kernel/kernel.o: external_kernel/kernel.cc
	$(MAKE) -C external_kernel mlir


kernel.o: external_kernel/kernel.o
	cp external_kernel/kernel.o ./kernel.o

kernel_matmul.o: matmul_kernel/kernel.o
	cp matmul_kernel/kernel.o ./kernel_matmul.o

# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
tutorial-9.exe: test.cpp aie.mlir kernel.o
	aiecc.py -j4 --sysroot=${SYSROOT} --host-target=aarch64-linux-gnu $(word 2,$^) -I$(AIE_INSTALL)/runtime_lib $(AIE_INSTALL)/runtime_lib/test_library.cpp ./$< -o $@

clean:
	$(MAKE) -C external_kernel clean
	rm -rf acdc_project *elf *exe core* kernel.o

#------------------------------------------------------------------------------
# Additional make targets for tutorial exercises
#------------------------------------------------------------------------------
tutorial-9_perf.exe: ./answers/test_perf.cpp ./aie.mlir kernel.o
	aiecc.py -j4 --sysroot=${SYSROOT} --host-target=aarch64-linux-gnu $(word 2,$^) -I$(AIE_INSTALL)/runtime_lib $(AIE_INSTALL)/runtime_lib/test_library.cpp ./$< -o $@

tutorial-9_matmul_perf.exe: ./answers/test_perf.cpp ./answers/aie_matmul.mlir kernel_matmul.o
	aiecc.py -j4 --sysroot=${SYSROOT} --host-target=aarch64-linux-gnu $(word 2,$^) -I$(AIE_INSTALL)/runtime_lib $(AIE_INSTALL)/runtime_lib/test_library.cpp ./$< -o $@
