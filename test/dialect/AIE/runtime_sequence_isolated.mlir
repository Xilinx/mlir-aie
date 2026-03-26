//===- runtime_sequence_isolated.mlir - IsolatedFromAbove --------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Tests that aie.runtime_sequence has the IsolatedFromAbove trait.
// Values defined inside the runtime_sequence and symbol references work fine.
// SCF ops inside runtime_sequence are preserved (not lowered by SCF→CF).
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: aie.runtime_sequence
// CHECK: scf.for
// CHECK: scf.if
// CHECK: arith.divui
// CHECK: aiex.npu.sync
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %rtp = aie.buffer(%tile_0_2) {sym_name = "rtp"} : memref<16xi32>

    // Runtime sequence with SSA params, SCF loops — all legal because
    // everything is defined INSIDE the IsolatedFromAbove region.
    aie.runtime_sequence(%in : memref<16xi32>, %out : memref<16xi32>, %n : i32) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c16 = arith.constant 16 : i32

      %n_div_16 = arith.divui %n, %c16 : i32

      %c0_idx = arith.index_cast %c0 : i32 to index
      %c1_idx = arith.index_cast %c1 : i32 to index
      %n_idx = arith.index_cast %n_div_16 : i32 to index

      %result = scf.for %i = %c0_idx to %n_idx step %c1_idx
          iter_args(%acc = %c0) -> (i32) {
        %i32 = arith.index_cast %i : index to i32
        %cmp = arith.cmpi sgt, %i32, %c0 : i32
        %val = scf.if %cmp -> (i32) {
          scf.yield %c1 : i32
        } else {
          scf.yield %c0 : i32
        }
        %new = arith.addi %acc, %val : i32
        scf.yield %new : i32
      }

      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}

// -----

// Negative test: using a value defined outside the runtime_sequence
// should be rejected because runtime_sequence is IsolatedFromAbove.
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %outside_val = arith.constant 42 : i32

    aie.runtime_sequence(%in : memref<16xi32>) {
      // expected-error @below {{using value defined outside the region}}
      %use = arith.addi %outside_val, %outside_val : i32
    }
  }
}
