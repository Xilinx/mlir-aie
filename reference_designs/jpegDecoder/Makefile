
.PHONY: all clean

# TODO - Change this to env variable?
SYSROOT = /group/xrlabs/platforms/vck190-pynq-v2.7/sysroot

buildDIR := ./buildAIE
VPATH := ./jpeg9reference/src ./src ./aieKernels

MY_VITIS_BIN_DIR := `dirname \`which vitis\``

all: aieObjects

aieKernelSrc := dequant.cc idct_horizontal.cc idct_vertical.cc
aieKernelObjs := $(aieKernelSrc:%=$(buildDIR)/%.o)

#external_kernel/work/Release_LLVM/test.prx/kernel.o: external_kernel/kernel.cc
#	$(MAKE) -C external_kernel build

jpegDecOnAIE: $(buildDIR)/jpegDecOnAIE.exe
aieObjects: $(aieKernelObjs)

$(buildDIR)/%.cc.o: %.cc
	xchesscc -p me -P ${MY_VITIS_BIN_DIR}/../aietools/data/cervino/lib/ -c $< -o $@


# Command line mlir-aie compile script "aiecc.py"
# Sysroot and host target used to cross compile  host executable
# Local runtime_lib needed for testbench functions
$(buildDIR)/jpegDecOnAIE.exe: test.cpp aie.mlir aieObjects
	aiecc.py -j4 --sysroot=${SYSROOT} --host-target=aarch64-linux-gnu $(word 2,$^) -I../../runtime_lib ../../runtime_lib/test_library.cpp ./$< -o $@

clean:
	rm -rf acdc_project *elf buildAIE/*exe buildAIE/*.cc.o core_7_*  
