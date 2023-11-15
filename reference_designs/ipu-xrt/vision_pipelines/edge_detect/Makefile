include ../makefile-common

build/aie.mlir: aie2_lineBased_8b_tiny.py
	mkdir -p ${@D}
	python3 $< > build/tmp.mlir
	aie-opt --canonicalize build/tmp.mlir -o $@
	rm build/tmp.mlir

clean:
	rm -rf build