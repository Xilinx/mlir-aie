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

.PHONY: all clean test

all: build/test

build/mv.o: $(KERNEL_DIR)/mv.cc | build
	$(PEANO_CLANG) $(CLANG_FLAGS) -c $< -o $@

build/mul.o: $(KERNEL_DIR)/mul.cc | build
	$(PEANO_CLANG) $(CLANG_FLAGS) -c $< -o $@

build/silu.o: $(KERNEL_DIR)/silu.cc | build
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

$(ELF_FILE): $(COMBINED_MLIR) build/mv.o build/mul.o build/silu.o
	cd build && $(AIECC) -v --generate-full-elf --no-xchesscc --no-xbridge aie.mlir -o aie.elf

build/test: test.cpp $(ELF_FILE)
	g++-13 $< -o $@ $(HOST_FLAGS)

build:
	mkdir -p build

test: build/test
	cd build && ./test

clean:
	rm -rf build aie.mlir.prj
