VPATH := /scratch/ehunhoff/mlir-aie/aie_kernels/generic

BUILD := build
MLIR := $(BUILD)/dei.mlir
XCLBIN := $(BUILD)/dei.xclbin
INSTS := $(BUILD)/dei_insts.bin
KERNEL_O := $(BUILD)/passThrough.cc.o

# Match yolo26n's PEANO flags shape.
PEANOWRAP2P_FLAGS := -O2 -std=c++20 --target=aie2p-none-unknown-elf \
    -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
    -Wno-missing-template-arg-list-after-template-kw \
    -DNDEBUG -DBIT_WIDTH=8 \
    -I $(MLIR_AIE_INSTALL_DIR)/include

.PHONY: all run clean
all: $(XCLBIN)

$(BUILD):
	mkdir -p $@

$(KERNEL_O): /scratch/ehunhoff/mlir-aie/aie_kernels/generic/passThrough.cc | $(BUILD)
	$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANOWRAP2P_FLAGS) -c $< -o $@

$(MLIR): test_dei.py | $(BUILD)
	python3 test_dei.py > $@

$(XCLBIN): $(MLIR) $(KERNEL_O)
	cd $(BUILD) && aiecc.py --aie-generate-xclbin --no-compile-host \
	    --xclbin-name=dei.xclbin --no-xchesscc --no-xbridge \
	    --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=dei_insts.bin \
	    ../$(MLIR)

run: $(XCLBIN)
	python3 run_dei.py -x $(XCLBIN) -i $(INSTS) -k MLIR_AIE

clean:
	rm -rf $(BUILD)
