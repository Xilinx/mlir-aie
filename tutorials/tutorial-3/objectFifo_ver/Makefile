
.PHONY: all clean

# TODO - Change this to env variable?
SYSROOT = /group/xrlabs/platforms/vck190-pynq-v2.7/sysroot

all: tutorial-3.exe

# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
tutorial-3.exe: aie.mlir test.cpp
	aiecc.py -j4 --sysroot=${SYSROOT} --host-target=aarch64-linux-gnu ./aie.mlir -I../../../runtime_lib ../../../runtime_lib/test_library.cpp ./test.cpp -o $@

Work/ps/c_rts/systemC/generated-objects/ps_i96.so: test.cpp aie.mlir
	make -C Work/ps/c_rts/systemC link

sim: Work/ps/c_rts/systemC/generated-objects/ps_i96.so
	aiesimulator --pkg-dir=./Work --dump-vcd foo

clean:
	rm -rf acdc_project *elf core* *log pl_sample_counts foo.vcd *exe aiesimulator_output

#------------------------------------------------------------------------------
# Additional make targets for tutorial exercises
#------------------------------------------------------------------------------
tutorial-3_perf.exe: ./answers/test_perf.cpp aie.mlir
	aiecc.py -j4 --sysroot=${SYSROOT} --host-target=aarch64-linux-gnu $(word 2,$^) -I../../runtime_lib ../../runtime_lib/test_library.cpp ./$< -o $@
