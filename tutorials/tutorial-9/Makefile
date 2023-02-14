
.PHONY: all clean

# TODO - Change this to env variable?
ifeq ($(MLIR_AIE_SYSROOT),)
MLIR_AIE_SYSROOT = /group/xrlabs/platforms/vck190-pynq-v2.7/sysroot
endif
AIE_INSTALL = $(dir $(shell which aie-opt))/..

all: tutorial-9.exe
	@mkdir -p elf
	@mv *.elf* ./elf
	@cp ./elf/*.elf ./elf/*.elf.map .

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
	aiecc.py -j4 --sysroot=${MLIR_AIE_SYSROOT} --host-target=aarch64-linux-gnu $(word 2,$^) -I$(AIE_INSTALL)/runtime_lib $(AIE_INSTALL)/runtime_lib/test_library.cpp ./$< -o $@

clean:
	$(MAKE) -C external_kernel clean
	rm -rf acdc_project sim aiesimulator_output *elf core* *log *vcd *exe pl_sample_counts *.o .AIE_SIM_CMD_LINE_OPTIONS

#------------------------------------------------------------------------------
# Additional make targets for tutorial exercises
#------------------------------------------------------------------------------
tutorial-9_perf.exe: ./answers/test_perf.cpp ./aie.mlir kernel.o
	aiecc.py -j4 --sysroot=${MLIR_AIE_SYSROOT} --host-target=aarch64-linux-gnu $(word 2,$^) -I$(AIE_INSTALL)/runtime_lib $(AIE_INSTALL)/runtime_lib/test_library.cpp ./$< -o $@

tutorial-9_matmul_perf.exe: ./answers/test_perf.cpp ./answers/aie_matmul.mlir kernel_matmul.o
	aiecc.py -j4 --sysroot=${MLIR_AIE_SYSROOT} --host-target=aarch64-linux-gnu $(word 2,$^) -I$(AIE_INSTALL)/runtime_lib $(AIE_INSTALL)/runtime_lib/test_library.cpp ./$< -o $@
