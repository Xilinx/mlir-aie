
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
tutorial-3.exe: aie.mlir test.cpp
	aiecc.py -j4 --sysroot=${MLIR_AIE_SYSROOT} --host-target=aarch64-linux-gnu $(word 2,$^) -I$(AIE_INSTALL)/runtime_lib $(AIE_INSTALL)/runtime_lib/test_library.cpp ./$< -o $@

Work/ps/c_rts/systemC/generated-objects/ps_i96.so: test.cpp aie.mlir
	make -C Work/ps/c_rts/systemC link

sim: Work/ps/c_rts/systemC/generated-objects/ps_i96.so
	aiesimulator --pkg-dir=./Work --dump-vcd foo

clean:
	rm -rf acdc_project sim aiesimulator_output *elf core* *log *vcd *exe pl_sample_counts

#------------------------------------------------------------------------------
# Additional make targets for tutorial exercises
#------------------------------------------------------------------------------
tutorial-3_perf.exe: ./answers/test_perf.cpp aie.mlir
	aiecc.py -j4 --sysroot=${MLIR_AIE_SYSROOT} --host-target=aarch64-linux-gnu $(word 2,$^) -I../../runtime_lib ../../runtime_lib/test_library.cpp ./$< -o $@
