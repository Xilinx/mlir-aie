//===- bank_aware_objectfifo.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verifier + lowering coverage for the per-tile bank-pinning attributes
// `producer_mem_bank` and `consumer_mem_banks` on aie.objectfifo.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectFifo-stateful-transform %s | FileCheck %s

// -----

// Happy path: producer_mem_bank is propagated to the producer-side BufferOp's
// mem_bank attribute by the stateful transform.

// CHECK-LABEL: module @producer_bank_propagates {
// CHECK:         aie.buffer({{.*}}) {mem_bank = 2 : i32, sym_name = "of_prod_buff_0"}
// CHECK:         aie.buffer({{.*}}) {mem_bank = 2 : i32, sym_name = "of_prod_buff_1"}
module @producer_bank_propagates {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.objectfifo @of_prod(%t02, {%t03}, 2 : i32) {producer_mem_bank = 2 : i32}
      : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// Verifier accepts a well-formed consumer_mem_banks attribute (one entry
// matching the single consumer tile). Whether the consumer-side buffer
// actually lands on the consumer tile depends on via_DMA / placement and is
// covered by the existing buffer-allocation tests; here we only verify the
// attribute is accepted end-to-end without crashing the transform.

// CHECK-LABEL: module @consumer_banks_accepted {
// CHECK:         aie.device(npu1_1col)
module @consumer_banks_accepted {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.objectfifo @of_cons(%t02, {%t03}, 2 : i32) {consumer_mem_banks = [1 : i32]}
      : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// Producer bank index >= numBanks(tile) is rejected at verifier time.
module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    // expected-error@+1 {{`producer_mem_bank` (7) out of range [0, 4) for tile (0, 2)}}
    aie.objectfifo @of_bad_prod(%t02, {%t03}, 2 : i32) {producer_mem_bank = 7 : i32}
      : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// consumer_mem_banks length mismatch (1 entry, 2 consumer tiles) is rejected.
module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    // expected-error@+1 {{`consumer_mem_banks` length must equal number of consumer tiles}}
    aie.objectfifo @of_bad_cons_len(%t02, {%t03, %t04}, 2 : i32)
        {consumer_mem_banks = [1 : i32]} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// A bad consumer bank index (out of range for that specific consumer tile)
// is reported with the offending index in the error message.
module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    // expected-error@+1 {{`consumer_mem_banks[1]` (9) out of range [0, 4) for tile (0, 4)}}
    aie.objectfifo @of_bad_cons_idx(%t02, {%t03, %t04}, 2 : i32)
        {consumer_mem_banks = [0 : i32, 9 : i32]} : !aie.objectfifo<memref<16xi32>>
  }
}
