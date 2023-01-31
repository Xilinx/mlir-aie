
.PHONY: all clean

SYSROOT = /group/xrlabs/platforms/vck190-pynq-v2.7/sysroot

all: test.elf
test.elf: aie.mlir test.cpp 
	aiecc.py -j4 --sysroot=${SYSROOT} --host-target=aarch64-linux-gnu ./aie.mlir -I../../../runtime_lib ../../../runtime_lib/test_library.cpp ./test.cpp -o $@

clean:
	rm -rf acdc_project *elf *exe core*
