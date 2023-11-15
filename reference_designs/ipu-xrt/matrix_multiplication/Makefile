include ../makefile-common

M?=128
K?=128
N?=128

m?=64
k?=32
n?=64

dtype_in?=i16
ifeq ($(dtype_in), i16)
	word_size_in?=2
	r?=4
	s?=4
	t?=4
	dtype_out?=i16
	word_size_out?=${word_size_in}
endif
ifeq ($(dtype_in), bf16)
	word_size_in?=2
	r?=4
	s?=8
	t?=4
	dtype_out?=f32
	word_size_out?=4
endif

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

build/final.xclbin: build/aie.mlir build/mm.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-ipu --ipu-insts-name=insts.txt $(<:%=../%)

build/test.exe: test.cpp
	mkdir -p ${@D}
	$(CXX) $(LD_FLAGS) $< -o $@ $(CFLAGS) $(INC) $(LD_PATHS) 

clean:
	rm -rf build
