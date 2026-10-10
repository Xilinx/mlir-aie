//===- bad_npu_nd_mixed_list_counts.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A static list whose kDynamic-sentinel count disagrees with its operand list must be
// DIAGNOSED, not asserted on. getMixedOffsets()/getMixedSizes() assert that the two agree,
// so any consumer of the mixed lists has to run after the count checks -- which is the order
// mlir::detail::verifyOffsetSizeAndStrideOp uses.
//
// Generic syntax on purpose: no well-formed producer emits this, so it takes hand-written or
// fuzzed input to reach. Before the reorder this aborted in
// mlir/lib/Dialect/Utils/StaticValueUtils.cpp (assertions build) rather than reaching any
// diagnostic at all.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      // static_sizes carries one kDynamic sentinel, but the $sizes operand list is empty.
      // expected-error@+1 {{expected 1 dynamic size values}}
      "aiex.npu.dma_memcpy_nd"(%arg0) <{
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        static_offsets = array<i64: 0, 0, 0, 0>,
        static_sizes = array<i64: -9223372036854775808, 4, 4, 4>,
        static_strides = array<i64: 0, 0, 0, 1>,
        metadata = @toMem,
        id = 1 : i64
      }> : (memref<16xi32>) -> ()
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @toMem (%tile_0_0, S2MM, 0)
  }
}
