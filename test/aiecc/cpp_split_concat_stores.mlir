//===- cpp_split_concat_stores.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A store of two conversions' concatenation reaches LLVM as a store of each
// conversion, not a shufflevector and one wide store. Only IR outputs are
// requested, so no core compiler is needed.

// RUN: %aiecc --get='llvmIR_{0}.ll' --tmpdir=%t --output-dir=%t %s
// RUN: FileCheck %s --input-file=%t/llvmIR_main_core_0_2.ll

// CHECK: %[[A:.*]] = call <16 x bfloat> @llvm.aie2p.v16accfloat.to.v16bf16
// CHECK: %[[B:.*]] = call <16 x bfloat> @llvm.aie2p.v16accfloat.to.v16bf16
// CHECK-NOT: shufflevector
// CHECK: store <16 x bfloat> %[[A]], ptr @out, align 2
// CHECK: store <16 x bfloat> %[[B]], ptr getelementptr inbounds{{.*}}(i8, ptr @out, i64 32), align 2

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %in = aie.buffer(%tile) {sym_name = "in"} : memref<32xf32>
    %out = aie.buffer(%tile) {sym_name = "out"} : memref<32xbf16>
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.0 : f32
      %a = vector.transfer_read %in[%c0], %cst {in_bounds = [true]} : memref<32xf32>, vector<16xf32>
      %b = vector.transfer_read %in[%c16], %cst {in_bounds = [true]} : memref<32xf32>, vector<16xf32>
      %ta = arith.truncf %a : vector<16xf32> to vector<16xbf16>
      %tb = arith.truncf %b : vector<16xf32> to vector<16xbf16>
      %c = vector.shuffle %ta, %tb [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xbf16>, vector<16xbf16>
      vector.transfer_write %c, %out[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16>
      aie.end
    }
  }
}
