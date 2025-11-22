KERNEL_NAMES = swiglu_gemv_1 swiglu_silu swiglu_eltwise_mul swiglu_gemv_2
MLIR_FILES = $(addprefix build/,$(addsuffix .mlir,$(KERNEL_NAMES)))
FINAL_XCLBIN = build/swiglu_combined.xclbin

PEANO_CLANG = $(PEANO_INSTALL_DIR)/bin/clang++
AIECC = aiecc.py
PEANO_PATH = $(PEANO_INSTALL_DIR)

KERNEL_DIR = ../../kernels
OPERATOR_DIR = ../../operators
AIE_KERNELS_DIR = /scratch/roesti/mlir-aie/aie_kernels
MLIR_AIE_DIR = $(shell python3 -c "from aie.utils.config import root_path; print(root_path())")

CLANG_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw -I$(AIE_KERNELS_DIR) -I$(MLIR_AIE_DIR)/include
AIECC_FLAGS = --no-compile-host --no-xchesscc --no-xbridge --peano $(PEANO_PATH)
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

build/swiglu_gemv_1.xclbin: build/swiglu_gemv_1.mlir build/mv.o
	cd build && $(AIECC) $(AIECC_FLAGS) --xclbin-instance-name=swiglu_gemv_1 --xclbin-kernel-id=0x901 --aie-generate-xclbin --xclbin-name=swiglu_gemv_1.xclbin --xclbin-kernel-name=swiglu_gemv_1 --aie-generate-npu --npu-insts-name=swiglu_gemv_1.bin swiglu_gemv_1.mlir

build/swiglu_silu.bin: build/swiglu_silu.mlir build/silu.o
	cd build && $(AIECC) $(AIECC_FLAGS) --no-compile --aie-generate-npu --npu-insts-name=swiglu_silu.bin swiglu_silu.mlir

build/swiglu_eltwise_mul.bin: build/swiglu_eltwise_mul.mlir build/mul.o
	cd build && $(AIECC) $(AIECC_FLAGS) --no-compile --aie-generate-npu --npu-insts-name=swiglu_eltwise_mul.bin swiglu_eltwise_mul.mlir

build/swiglu_gemv_2.bin: build/swiglu_gemv_2.mlir build/mv.o
	cd build && $(AIECC) $(AIECC_FLAGS) --no-compile --aie-generate-npu --npu-insts-name=swiglu_gemv_2.bin swiglu_gemv_2.mlir

build/swiglu_silu.xclbin: build/swiglu_silu.mlir build/swiglu_gemv_1.xclbin build/swiglu_silu.bin
	cd build && $(AIECC) $(AIECC_FLAGS) --xclbin-instance-name=swiglu_silu --xclbin-kernel-id=0x902 --aie-generate-xclbin --xclbin-name=swiglu_silu.xclbin --xclbin-kernel-name=swiglu_silu --xclbin-input=swiglu_gemv_1.xclbin swiglu_silu.mlir

build/swiglu_eltwise_mul.xclbin: build/swiglu_eltwise_mul.mlir build/swiglu_silu.xclbin build/swiglu_eltwise_mul.bin
	cd build && $(AIECC) $(AIECC_FLAGS) --xclbin-instance-name=swiglu_eltwise_mul --xclbin-kernel-id=0x903 --aie-generate-xclbin --xclbin-name=swiglu_eltwise_mul.xclbin --xclbin-kernel-name=swiglu_eltwise_mul --xclbin-input=swiglu_silu.xclbin swiglu_eltwise_mul.mlir

build/swiglu_combined.xclbin: build/swiglu_gemv_2.mlir build/swiglu_eltwise_mul.xclbin build/swiglu_gemv_2.bin
	cd build && $(AIECC) $(AIECC_FLAGS) --xclbin-instance-name=swiglu_gemv_2 --xclbin-kernel-id=0x904 --aie-generate-xclbin --xclbin-name=swiglu_combined.xclbin --xclbin-kernel-name=swiglu_gemv_2 --xclbin-input=swiglu_eltwise_mul.xclbin swiglu_gemv_2.mlir

build/test: test.cpp $(FINAL_XCLBIN)
	g++-13 $< -o $@ $(HOST_FLAGS)

build:
	mkdir -p build

test: build/test
	cd build && ./test

clean:
	rm -rf build aie.mlir.prj
