srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../makefile-common

targetname = compression_check
LENGTH ?= 4096
CONFIG  ?= base

# Symmetric / unsized configs (build dir == config name).
SYM_CONFIGS = base cmp_only dcmp_only both

# Ratioed asymmetric configs: tag, base config, shim_in_n, shim_out_n.
# Sized so the shim BD lengths match the empirical compressed byte count
# for arange input (ratio ~1.39x for ramp 0..N-1) so the consumer DMA does
# not hang waiting for bytes that never arrive.
RATIO_CONFIGS = cmp_only_ratioed dcmp_only_ratioed
cmp_only_ratioed_BASE  = cmp_only
cmp_only_ratioed_IN_N  = 4096
cmp_only_ratioed_OUT_N = 2944
dcmp_only_ratioed_BASE  = dcmp_only
dcmp_only_ratioed_IN_N  = 2944
dcmp_only_ratioed_OUT_N = 4096

ALL_CONFIGS = $(SYM_CONFIGS) $(RATIO_CONFIGS)

all: build/$(CONFIG)/final.xclbin

# Symmetric build rule: in_n = out_n = LENGTH (default).
build/base/aie.mlir build/cmp_only/aie.mlir build/dcmp_only/aie.mlir build/both/aie.mlir: build/%/aie.mlir: ${srcdir}/$(targetname).py
	mkdir -p ${@D}
	python3 $< $* $(LENGTH) $(LENGTH) > $@

# Ratioed asymmetric build rule: pulls in_n / out_n from the per-config vars above.
$(addsuffix /aie.mlir,$(addprefix build/,$(RATIO_CONFIGS))): build/%/aie.mlir: ${srcdir}/$(targetname).py
	mkdir -p ${@D}
	python3 $< $($*_BASE) $($*_IN_N) $($*_OUT_N) > $@

build/%/final.xclbin: build/%/aie.mlir
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
	./$< -x build/$(CONFIG)/final.xclbin -i build/$(CONFIG)/insts.bin -k MLIR_AIE -l $(LENGTH) -c $(CONFIG)

# Symmetric configs only — useful for first-pass triage.
run_sym: $(targetname).exe $(addsuffix /final.xclbin,$(addprefix build/,$(SYM_CONFIGS)))
	@echo "=========================================="
	@echo "Symmetric configs (LENGTH=$(LENGTH))"
	@echo "=========================================="
	@for c in $(SYM_CONFIGS); do \
		echo; echo "--- config: $$c ---"; \
		./$(targetname).exe -x build/$$c/final.xclbin -i build/$$c/insts.bin \
			-k MLIR_AIE -l $(LENGTH) -c $$c || echo "(returned $$?)"; \
	done

# Ratioed asymmetric configs — proof that compression itself is fast.
run_ratioed: $(targetname).exe $(addsuffix /final.xclbin,$(addprefix build/,$(RATIO_CONFIGS)))
	@echo "=========================================="
	@echo "Ratioed asymmetric configs"
	@echo "=========================================="
	@for c in $(RATIO_CONFIGS); do \
		echo; echo "--- config: $$c ---"; \
		./$(targetname).exe -x build/$$c/final.xclbin -i build/$$c/insts.bin \
			-k MLIR_AIE -l $(LENGTH) -c $$c || echo "(returned $$?)"; \
	done

test_all: run_sym run_ratioed
	@echo
	@echo "=========================================="
	@echo "Done. See README.md for expected outputs."
	@echo "=========================================="

clean:
	rm -rf build _build $(targetname).exe
