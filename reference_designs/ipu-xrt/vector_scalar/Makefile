include ../makefile-common

all: build/final.xclbin build/insts.txt

run: all test.exe
	pushd build; ./test.exe -x ./final.xclbin -k MLIR_AIE -i ./insts.txt; popd

build/%.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc ${CHESSCC2_FLAGS} -DBIT_WIDTH=8 -c $(<:%=../%) -o ${@F}

build/aie.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< > build/tmp.mlir
	aie-opt --canonicalize build/tmp.mlir -o $@
	rm build/tmp.mlir

build/final.xclbin: build/aie.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-ipu --ipu-insts-name=insts.txt $(<:%=../%)

build/test.exe: test.cpp
	mkdir -p ${@D}
	$(CXX) $(LD_FLAGS) $< -o $@ $(CFLAGS) $(INC) $(LD_PATHS) 

clean:
	rm -rf build