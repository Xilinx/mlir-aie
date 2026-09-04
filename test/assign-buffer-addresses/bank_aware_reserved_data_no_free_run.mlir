//===- bank_aware_reserved_data_no_free_run.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The boundary case for largestFreeRun(): a tile whose stack and one buffer
// take every byte, so the largest free run is exactly 0. reserved_data_size = 0
// succeeds, because 0 bytes fit a 0-byte run. Any larger reservation fails; see
// bank_aware_reserved_data_no_free_run_error.mlir.
//
// tile(0, 2) on npu2 has 65536 bytes. The 1024-byte stack plus "a"'s 64512
// unaligned bytes cover all of them.

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// CHECK: %a = aie.buffer(%tile_0_2) {address = 1024 : i32, aligned = false, mem_bank = 0 : i32, sym_name = "a"} : memref<64512xi8>
// No space remains, so the granted region is empty, and reserved_data_size is 0
// here, so the empty grant satisfies the request. A zero-length region spans no
// bytes, so the verifier's placement check skips it; a placement check would
// read data_origin = 0 as a region below the stack.
// CHECK: data_length = 0 : i32, data_origin = 0 : i32
module @zero_free_run_zero_reservation {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %a = aie.buffer(%tile_0_2) {sym_name = "a", aligned = false} : memref<64512xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, reserved_data_size = 0 : i32}
  }
}
