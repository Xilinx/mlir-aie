MLIR_FILES = build/swiglu_gemv_1.mlir build/swiglu_silu.mlir build/swiglu_eltwise_mul.mlir build/swiglu_gemv_2.mlir
COMBINED_MLIR = build/aie.mlir
ELF_FILE = build/aie.elf

AIECC = aiecc.py
PEANO_CLANG = $(PEANO_INSTALL_DIR)/bin/clang++
PEANO_PATH = $(PEANO_INSTALL_DIR)

KERNEL_DIR = ../../kernels
OPERATOR_DIR = ../../operators
AIE_KERNELS_DIR = /scratch/roesti/mlir-aie/aie_kernels
MLIR_AIE_DIR = $(shell python3 -c "from aie.utils.config import root_path; print(root_path())")

CLANG_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw -I$(AIE_KERNELS_DIR) -I$(MLIR_AIE_DIR)/include
HOST_FLAGS = -std=c++23 -Wall -I/opt/xilinx/xrt/include -I/scratch/roesti/mlir-aie/runtime_lib/test_lib -L/opt/xilinx/xrt/lib -lxrt_core -lxrt_coreutil -lrt -lstdc++

VARIANT_NAME?=fused_transactions_write32s

.PHONY: all clean test all_reset_modes build_reset_always build_reset_ifused build_reset_ifchanged build_reset_ifchangedfinegrained build_reset_never

all: build/test

build/mv.o: $(KERNEL_DIR)/mv.cc | build
	$(PEANO_CLANG) $(CLANG_FLAGS) -c $< -o $@

build/mul.o: $(KERNEL_DIR)/mul.cc | build
	$(PEANO_CLANG) $(CLANG_FLAGS) -c $< -o $@

build/silu.o: $(KERNEL_DIR)/silu.cc | build
	$(PEANO_CLANG) $(CLANG_FLAGS) -c $< -o $@

$(BUILD_DIR)/mv.o: $(KERNEL_DIR)/mv.cc | $(BUILD_DIR)
	$(PEANO_CLANG) $(CLANG_FLAGS) -c $< -o $@

$(BUILD_DIR)/mul.o: $(KERNEL_DIR)/mul.cc | $(BUILD_DIR)
	$(PEANO_CLANG) $(CLANG_FLAGS) -c $< -o $@

$(BUILD_DIR)/silu.o: $(KERNEL_DIR)/silu.cc | $(BUILD_DIR)
	$(PEANO_CLANG) $(CLANG_FLAGS) -c $< -o $@

build/swiglu_gemv_1.mlir: $(OPERATOR_DIR)/matrix_vector_mul.py | build
	python3 -c 'import sys; sys.path.append("$(OPERATOR_DIR)"); from matrix_vector_mul import my_matvec; from aie.iron.device import NPU2; mlir_code = my_matvec(NPU2(), 8, 8192, 2048, 1); print(str(mlir_code))' > $@

build/swiglu_gemv_2.mlir: $(OPERATOR_DIR)/matrix_vector_mul.py | build
	python3 -c 'import sys; sys.path.append("$(OPERATOR_DIR)"); from matrix_vector_mul import my_matvec; from aie.iron.device import NPU2; mlir_code = my_matvec(NPU2(), 8, 2048, 8192, 1); print(str(mlir_code))' > $@

build/swiglu_silu.mlir: $(OPERATOR_DIR)/silu.py | build
	python3 -c 'import sys; sys.path.append("$(OPERATOR_DIR)"); from silu import my_silu; from aie.iron.device import NPU2; mlir_code = my_silu(NPU2(), 8192, 8, 2, 512, 0); print(str(mlir_code))' > $@

build/swiglu_eltwise_mul.mlir: $(OPERATOR_DIR)/eltwise_mul.py | build
	python3 -c 'import sys; sys.path.append("$(OPERATOR_DIR)"); from eltwise_mul import my_eltwise_mul; from aie.iron.device import NPU2; mlir_code = my_eltwise_mul(NPU2(), 8192, 8, 2, 1024, 0); print(str(mlir_code))' > $@

$(COMBINED_MLIR): $(MLIR_FILES) combine_mlir.py
	python3 combine_mlir.py $(MLIR_FILES) $(COMBINED_MLIR)

RESET_MODE?=always
BUILD_DIR?=build

$(BUILD_DIR)/aie.elf: $(COMBINED_MLIR) $(BUILD_DIR)/mv.o $(BUILD_DIR)/mul.o $(BUILD_DIR)/silu.o | $(BUILD_DIR)
	cp $(COMBINED_MLIR) $(BUILD_DIR)/aie.mlir
	cd $(BUILD_DIR) && $(AIECC) -v --no-coalesce-write32s --generate-full-elf --expand-load-pdis --reset-mode=$(RESET_MODE) --reset-tiles=core,mem --no-xchesscc --no-xbridge aie.mlir -o aie.elf

$(BUILD_DIR)/test: test.cpp $(BUILD_DIR)/aie.elf
	g++-13 $< -o $@ $(HOST_FLAGS) -DVARIANT_NAME=\"$(VARIANT_NAME)\"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

build:
	mkdir -p build

# Build all reset mode variants
all_reset_modes: _build_reset_always _build_reset_ifused _build_reset_ifchanged _build_reset_ifchangedfinegrained _build_reset_never

_build_reset_always:
	$(MAKE) BUILD_DIR=build_reset_always RESET_MODE=always VARIANT_NAME=fused_write32s_reset_always build_reset_always/test

_build_reset_ifused:
	$(MAKE) BUILD_DIR=build_reset_ifused RESET_MODE=ifused VARIANT_NAME=fused_write32s_reset_ifused build_reset_ifused/test

_build_reset_ifchanged:
	$(MAKE) BUILD_DIR=build_reset_ifchanged RESET_MODE=ifchanged VARIANT_NAME=fused_write32s_reset_ifchanged build_reset_ifchanged/test

_build_reset_ifchangedfinegrained:
	$(MAKE) BUILD_DIR=build_reset_ifchangedfinegrained RESET_MODE=ifchangedfinegrained VARIANT_NAME=fused_write32s_reset_ifchangedfinegrained build_reset_ifchangedfinegrained/test

_build_reset_never:
	$(MAKE) BUILD_DIR=build_reset_never RESET_MODE=never VARIANT_NAME=fused_write32s_reset_never build_reset_never/test

test: $(BUILD_DIR)/test
	cd $(BUILD_DIR) && ./test

clean:
	rm -rf build build_reset_* aie.mlir.prj
