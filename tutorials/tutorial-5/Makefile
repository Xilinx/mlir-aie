
.PHONY: all clean

# TODO - Change this to env variable?
SYSROOT = /group/xrlabs/platforms/vck190-pynq-v2.7/sysroot

all: tutorial-5.exe

# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
tutorial-5.exe: aie.mlir test.cpp 
	aiecc.py --sysroot=${SYSROOT} --host-target=aarch64-linux-gnu ./aie.mlir -I../../runtime_lib ../../runtime_lib/test_library.cpp ./test.cpp -o $@

view: routes.json

routes.json: aie.mlir
	aie-opt --aie-create-pathfinder-flows --aie-find-flows $< | aie-translate --aie-flows-to-json > $@

clean:
	rm -rf acdc_project *elf *exe core*
