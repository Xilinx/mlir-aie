//===- cpp_aie2p_vector_add.mlir - Regression test for #2950 ------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Regression test for #2950: C++ aiecc must correctly propagate the aie-target
// option to convert-vector-to-aievec sub-pipelines. Without the fix, aie2p
// targets fall back to aie1 lowering patterns (producing aievec_aie1 ops),
// which causes aie-standard-lowering to fail to extract core functions.

// REQUIRES: peano

// RUN: aiecc --no-xchesscc --no-xbridge --verbose %s | FileCheck %s

// CHECK: LLVM lowering pipeline completed successfully
// CHECK: Compilation completed successfully

module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %buf_a = aie.buffer(%tile_0_2) {sym_name = "buf_a"} : memref<256xbf16>
    %buf_b = aie.buffer(%tile_0_2) {sym_name = "buf_b"} : memref<256xbf16>
    %buf_c = aie.buffer(%tile_0_2) {sym_name = "buf_c"} : memref<256xbf16>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.0 : bf16

      // Vector add loop: this exercises convert-vector-to-aievec.
      // With the correct aie2p target, this produces aievec.ups + aievec.add_elem + aievec.srs.
      // With the broken aie1 fallback, it produces aievec_aie1.add which can't be
      // lowered to LLVM, resulting in empty core function and linker failure.
      scf.for %i = %c0 to %c256 step %c16 {
        %a = vector.transfer_read %buf_a[%i], %cst {in_bounds = [true]} : memref<256xbf16>, vector<16xbf16>
        %b = vector.transfer_read %buf_b[%i], %cst {in_bounds = [true]} : memref<256xbf16>, vector<16xbf16>
        %c = arith.addf %a, %b : vector<16xbf16>
        vector.transfer_write %c, %buf_c[%i] {in_bounds = [true]} : vector<16xbf16>, memref<256xbf16>
      }
      aie.end
    }
  }
}
