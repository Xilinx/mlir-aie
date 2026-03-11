//===- cpp_f32_relu.mlir - Regression test for #2945 ------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Regression test for #2945: convert-to-llvm must use dynamic=true so that
// DataLayoutAnalysis configures the type converter correctly. Without it,
// modules with a data layout that sets index != 64 bits produce
// unrealized_conversion_cast ops that fail LLVM translation.

// REQUIRES: peano

// RUN: aiecc --no-xchesscc --no-xbridge --verbose %s | FileCheck %s

// CHECK: LLVM lowering pipeline completed successfully
// CHECK: Compilation completed successfully

module attributes {dlti.dl_spec = #dlti.dl_spec<index = 32>} {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %buf_in = aie.buffer(%tile_0_2) {sym_name = "buf_in"} : memref<256xf32>
    %buf_out = aie.buffer(%tile_0_2) {sym_name = "buf_out"} : memref<256xf32>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %cst = arith.constant 0.0 : f32

      // ReLU with f32 data and index loop variable.
      // Without convert-to-llvm{dynamic=true}, the index<->i64 mismatch
      // from the 32-bit index data layout causes unrealized_conversion_cast.
      scf.for %i = %c0 to %c256 step %c1 {
        %val = memref.load %buf_in[%i] : memref<256xf32>
        %gt = arith.cmpf ogt, %val, %cst : f32
        %relu = arith.select %gt, %val, %cst : f32
        memref.store %relu, %buf_out[%i] : memref<256xf32>
      }
      aie.end
    }
  }
}
