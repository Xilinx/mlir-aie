include ../../../utils/vitisVariables.config
include ../../../tutorials/makefile-common

.PHONY: all clean

all: host_nested_loop.exe
host_nested_loop.exe: test.cpp aie.mlir 
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

#	aiecc.py -j4 --sysroot=${SYSROOT} --host-target=aarch64-linux-gnu ./aie.mlir -I$(AIE_INSTALL)/runtime_lib $(AIE_INSTALL)/runtime_lib/test_library.cpp ./test.cpp -o $@


clean:
	rm -rf acdc_project *elf *exe core* sim
