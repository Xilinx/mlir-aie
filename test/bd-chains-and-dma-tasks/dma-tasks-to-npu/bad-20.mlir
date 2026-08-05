//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics %s

// A dma_bd nested inside a DMAConfigureTaskOp is exempt from AIE::DMABDOp::
// verify()'s own sizes/strides consistency check: that check only runs when
// the BD's parent is a MemOp/MemTileDMAOp/ShimDMAOp/DMAOp (see the comment at
// the top of DMABDOp::verify()), and a DMA-task BD's parent is none of those.
// verifyTaskBDDimensions() (AIEXDialect.cpp) then calls bd.getMixedSizes() on
// this never-validated BD to enforce the per-tile dimension cap. Before the
// guard this test exercises, that crashed instead of diagnosing a malformed
// static_sizes/sizes operand-count mismatch, because getMixedSizes() routes
// through mlir::getMixedValues(), which indexes past the end of the dynamic
// operand range on a mismatch (asserts in a debug build, OOB read otherwise).
//
// This state -- a static_sizes array whose dynamic-marker count disagrees
// with the number of `sizes` SSA operands actually supplied -- is not
// reachable through the pretty `sizes = [...]` syntax (its custom parser
// always keeps the two lists consistent), so this is hand-written via the
// generic op syntax to model malformed/fuzzed IR, not IR any real producer
// emits.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<32xi8>) {
      %t1 = "aiex.dma_configure_task"(%tile_0_0) <{channel = 0 : i32, direction = 1 : i32, operandSegmentSizes = array<i32: 1, 0, 0>}> ({
        // static_sizes declares one dynamic (kDynamic-sentinel) entry, but no
        // `sizes` operand backs it (operandSegmentSizes' 4th entry, for
        // $sizes, is 0).
        // expected-error@+1 {{expected 1 dynamic sizes values}}
        "aie.dma_bd"(%arg0) <{bd_id = 0 : i32, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, static_len = 32 : i32, static_offset = 4 : i32, static_sizes = array<i64: -9223372036854775808, 2>, static_strides = array<i64: 4, 1>}> : (memref<32xi8>) -> ()
        aie.end
      }) : (index) -> index
    }
  }
}
