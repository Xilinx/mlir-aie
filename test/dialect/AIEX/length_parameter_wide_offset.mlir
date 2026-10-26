//===- length_parameter_wide_offset.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A length_parameter BD whose runtime offset is wider than 64 bits, with a
// range that does not fit in int64_t. The extent is treated as unknown, so the
// parameter's bound falls back to the BD length field's limit.

// RUN: aie-opt --aie-lower-scratchpad-parameters %s | FileCheck %s

// CHECK: aiex.scratchpad_parameter @m : i32 {kind = 1 : i32, max_value = 134217726 : i32, min_value = 0 : i32
aiex.scratchpad_parameter @m : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence @wide(%arg0 : memref<4096xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %big = arith.constant 18446744073709551616 : i128
    scf.for %i = %c0 to %c2 step %c1 {
      %ii = arith.index_cast %i : index to i128
      %off = arith.muli %ii, %big : i128
      %task = aiex.dma_configure_task(%t, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4096xbf16> offset = %off : i128 len = 64) {length_parameter = @m, length_unit = 64 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
      aiex.dma_await_task(%task)
    }
  }
}
