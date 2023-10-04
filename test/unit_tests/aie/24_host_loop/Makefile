
.PHONY: all clean

SYSROOT = /group/xrlabs/platforms/vck190-pynq-v2.7/sysroot
AIE_INSTALL = $(dir $(shell which aie-opt))/..

all: test.elf
test.elf: aie.mlir test.cpp
	aiecc.py -j4 --sysroot=${SYSROOT} --host-target=aarch64-linux-gnu ./aie.mlir -I$(AIE_INSTALL)/runtime_lib $(AIE_INSTALL)/runtime_lib/test_library.cpp ./test.cpp -o $@

clean:
	rm -rf aie.mlir.prj *elf *exe core*
