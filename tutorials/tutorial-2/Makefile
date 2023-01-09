
.PHONY: all clean

# TODO - Change this to env variable?
SYSROOT = /group/xrlabs/platforms/vck190-pynq-v2.7/sysroot

all: tutorial-2.exe

#external_kernel/work/Release_LLVM/test.prx/kernel.o: external_kernel/kernel.cc
#	$(MAKE) -C external_kernel build

external_kernel/kernel.o: external_kernel/kernel.cc
	$(MAKE) -C external_kernel mlir


kernel.o: external_kernel/kernel.o
	cp external_kernel/kernel.o .

# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
tutorial-2.exe: test.cpp aie.mlir kernel.o
	aiecc.py -j4 --sysroot=${SYSROOT} --host-target=aarch64-linux-gnu $(word 2,$^) -I../../runtime_lib ../../runtime_lib/test_library.cpp ./$< -o $@

clean:
	$(MAKE) -C external_kernel clean
	rm -rf acdc_project *elf *exe core* kernel.o
