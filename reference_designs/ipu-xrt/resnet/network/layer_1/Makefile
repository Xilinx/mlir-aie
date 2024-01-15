include ../../../makefile-common

#mlirFileName = bottlenck4AIEs
mlirFileName = aieWithTrace
# mlirFileName = aieFinal
# mlirFileName = aieDemo
all: conv2dk1_init_i8.o conv2dk1_skip_init_i8.o conv2dk1_ui8.o conv2dk3_ui8.o conv2dk1_skip_ui8.o final.xclbin
VPATH := ../../kernels

insts.txt: ${mlirFileName}.mlir
	aiecc.py -v --aie-only-generate-ipu --ipu-insts-name=$@ $<

conv2dk1_init_i8.o: scalar_conv2dk1_init_i8.cc
	xchesscc -d ${CHESSCC2_FLAGS} -c $< -o $@

conv2dk1_skip_init_i8.o: scalar_conv2dk1_skip_init_i8.cc
	xchesscc -d ${CHESSCC2_FLAGS} -c $< -o $@

conv2dk1_ui8.o: scalar_conv2dk1_ui8.cc
	xchesscc -d ${CHESSCC2_FLAGS} -c $< -o $@
	
conv2dk3_ui8.o: scalar_conv2dk3_ui8.cc
	xchesscc -d ${CHESSCC2_FLAGS} -c $< -o $@

conv2dk1_skip_ui8.o: scalar_conv2dk1_skip_ui8.cc
	xchesscc -d ${CHESSCC2_FLAGS} -c $< -o $@

final.xclbin: ${mlirFileName}.mlir 
	aiecc.py -v --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=$@ --ipu-insts-name=insts.txt $<

clean:
	rm -rf *.elf* *.lst *.bif *.mlir.prj *.xclbin sim \
		chess* *.o insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe

test.exe: test.cpp
	$(CXX) $(LD_FLAGS) $< -o $@ $(CFLAGS) $(INC) $(LD_PATHS) 

