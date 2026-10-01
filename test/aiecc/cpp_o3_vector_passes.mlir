//===- cpp_o3_vector_passes.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// -O3 adds aie-hoist-vector-transfer-pointers and aie-vector-to-pointer-loops
// to the input_with_symbols pipeline and aievec-split-load-ups-chains to the
// core lowering, for both per-core and unified lowering. Lower levels run none
// of them. Only IR outputs are requested, so no core compiler is needed.

// RUN: %aiecc --get=input_with_symbols.mlir --get='perCoreArches_{0}.txt' --get='llvmIR_{0}.ll' --tmpdir=%t.default --output-dir=%t.default %s
// RUN: FileCheck %s --check-prefix=NOVEC-IR --input-file=%t.default/input_with_symbols.mlir
// RUN: FileCheck %s --check-prefix=NOVEC-LL --input-file=%t.default/llvmIR_main_core_0_2.ll

// RUN: %aiecc -O2 --get=input_with_symbols.mlir --get='perCoreArches_{0}.txt' --get='llvmIR_{0}.ll' --tmpdir=%t.o2 --output-dir=%t.o2 %s
// RUN: FileCheck %s --check-prefix=NOVEC-IR --input-file=%t.o2/input_with_symbols.mlir
// RUN: FileCheck %s --check-prefix=NOVEC-LL --input-file=%t.o2/llvmIR_main_core_0_2.ll

// RUN: %aiecc -O2 --unified --get='perCoreArches_{0}.txt' --get='llvmIR_{0}.ll' --tmpdir=%t.o2u --output-dir=%t.o2u %s
// RUN: FileCheck %s --check-prefix=NOVEC-LL --input-file=%t.o2u/llvmIR_main_core_0_2.ll

// RUN: %aiecc -O3 --get=input_with_symbols.mlir --get='perCoreArches_{0}.txt' --get='llvmIR_{0}.ll' --tmpdir=%t.o3 --output-dir=%t.o3 %s
// RUN: FileCheck %s --check-prefix=VEC-IR --input-file=%t.o3/input_with_symbols.mlir
// RUN: FileCheck %s --check-prefix=VEC-LL --input-file=%t.o3/llvmIR_main_core_0_2.ll

// RUN: %aiecc -O3 --unified --get='perCoreArches_{0}.txt' --get='llvmIR_{0}.ll' --tmpdir=%t.o3u --output-dir=%t.o3u %s
// RUN: FileCheck %s --check-prefix=VEC-LL --input-file=%t.o3u/llvmIR_main_core_0_2.ll

// NOVEC-IR-LABEL: aie.core
// NOVEC-IR-NOT: memref.collapse_shape
// NOVEC-IR-NOT: ptr.
// NOVEC-IR: vector.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<16x64xi16>, vector<64xi16>
// NOVEC-IR: vector.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<16x64xi32>, vector<64xi32>
// NOVEC-IR-NOT: ptr.
// NOVEC-IR: aie.end

// Without the split, the whole v64xi16 is loaded and shuffled into halves
// before the ups.
// NOVEC-LL: shufflevector
// NOVEC-LL: call {{.*}}@llvm.aie2p.acc32.v32.I512.ups

// VEC-IR-LABEL: aie.core
// VEC-IR-DAG: memref.collapse_shape %{{.*}} {{\[\[}}0, 1]] : memref<16x64xi16> into memref<1024xi16>
// VEC-IR-DAG: memref.collapse_shape %{{.*}} {{\[\[}}0, 1]] : memref<16x64xi32> into memref<1024xi32>
// VEC-IR-DAG: ptr.to_ptr %{{.*}} : memref<1024xi16, #ptr.generic_space>
// VEC-IR-DAG: ptr.to_ptr %{{.*}} : memref<1024xi32, #ptr.generic_space>
// VEC-IR: vector.load %{{.*}}[%c0] : memref<1024xi16, #ptr.generic_space>, vector<64xi16>
// VEC-IR: ptr.ptr_add %{{.*}}, %c128 :
// VEC-IR: vector.store %{{.*}}, %{{.*}}[%c0] : memref<1024xi32, #ptr.generic_space>, vector<64xi32>
// VEC-IR: ptr.ptr_add %{{.*}}, %c256 :

// With the split, the two halves are loaded separately and one shuffle
// concatenates the ups results.
// VEC-LL-NOT: shufflevector
// VEC-LL: call {{.*}}@llvm.aie2p.acc32.v32.I512.ups
// VEC-LL-NOT: shufflevector
// VEC-LL: call {{.*}}@llvm.aie2p.acc32.v32.I512.ups
// VEC-LL: shufflevector
// VEC-LL-NOT: shufflevector

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %in = aie.buffer(%tile) {sym_name = "in"} : memref<16x64xi16>
    %out = aie.buffer(%tile) {sym_name = "out"} : memref<16x64xi32>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %pad = arith.constant 0 : i16
      scf.for %i = %c0 to %c16 step %c1 {
        %v = vector.transfer_read %in[%i, %c0], %pad {in_bounds = [true]} : memref<16x64xi16>, vector<64xi16>
        %e = arith.extsi %v : vector<64xi16> to vector<64xi32>
        vector.transfer_write %e, %out[%i, %c0] {in_bounds = [true]} : vector<64xi32>, memref<16x64xi32>
      }
      aie.end
    }
  }
}
