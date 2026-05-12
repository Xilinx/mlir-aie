srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../makefile-common

targetname = compression_check
LENGTH ?= 4096
CONFIG  ?= base

all: build/$(CONFIG)/final.xclbin

build/$(CONFIG)/aie.mlir: ${srcdir}/$(targetname).py
	mkdir -p ${@D}
	python3 $< $(CONFIG) > $@

build/$(CONFIG)/final.xclbin: build/$(CONFIG)/aie.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc --aie-generate-xclbin --xclbin-name=${@F} \
		--no-xchesscc --no-xbridge \
		--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../../%)

$(targetname).exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && cmake ${srcdir} -DTARGET_NAME=$(targetname)
	cd _build && cmake --build . --config Release
	cp _build/$(targetname) $@

run: $(targetname).exe build/$(CONFIG)/final.xclbin
	./$< -x build/$(CONFIG)/final.xclbin -i build/$(CONFIG)/insts.bin -k MLIR_AIE -l ${LENGTH} -c $(CONFIG)

run_all: $(targetname).exe build/base/final.xclbin build/cmp_only/final.xclbin build/dcmp_only/final.xclbin build/both/final.xclbin
	@echo "=========================================="
	@echo "Running all four configs (LENGTH=${LENGTH})"
	@echo "=========================================="
	@for c in base cmp_only dcmp_only both; do \
		echo; echo "--- config: $$c ---"; \
		./$(targetname).exe -x build/$$c/final.xclbin -i build/$$c/insts.bin \
			-k MLIR_AIE -l ${LENGTH} -c $$c || echo "(returned $$?)"; \
	done

clean:
	rm -rf build _build $(targetname).exe
