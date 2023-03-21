include ../../../utils/vitisVariables.config
include ../../../tutorials/makefile-common

.PHONY: all clean

all: host_multirate.exe
host_multirate.exe: test.cpp aie.mlir 
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

clean:
	rm -rf acdc_project *elf *exe core* sim
