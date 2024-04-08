include ../../../tutorials/makefile-common

.PHONY: all clean

all: test.elf
	@mkdir -p elf
	@mv *.elf* ./elf
	@cp ./elf/*.elf ./elf/*.elf.map .

kernel.o: ../kernel.cc
	xchesscc ${CHESSCC_FLAGS} -c $<

test.elf: test.cpp aie.mlir kernel.o
	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

clean:
	rm -rf aie.mlir.prj aiesimulator_output *elf core* *log *vcd *exe pl_sample_counts *.o .AIE_SIM_CMD_LINE_OPTIONS
