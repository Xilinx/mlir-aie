AIE_RUNTIME_LIB ?= $(shell realpath $(dir $(shell which aie-opt))/../runtime_lib)

.PHONY: all build/test clean

all: build/test

build/test : test.cpp aie.mlir
	mkdir -p ${@D}
	cd ${@D} && \
		aiecc.py -j4 --xchesscc --xbridge --aiesim ../$(word 2,$^) \
			-ggdb -I${AIE_RUNTIME_LIB}/x86_64/test_lib/include \
			-L${AIE_RUNTIME_LIB}/x86_64/test_lib/lib -ltest_lib \
			../$<

clean:
	rm -rf build