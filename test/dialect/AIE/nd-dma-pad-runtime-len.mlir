//===- nd-dma-pad-runtime-len.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

// Regression: a padded memtile BD with a runtime len operand must produce a
// clear "not yet supported" diagnostic rather than silently skipping the
// padding-exceeds-len check.

module {
  aie.device(xcve2802) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<256xi32>
    %mem = aie.memtile_dma(%t1) {
      %c2_i32 = arith.constant 2 : i32
      %c3_i32 = arith.constant 3 : i32
      // Use muli so len is not a compile-time constant in the SSA sense
      // (it's a non-trivially-foldable op result, simulating a runtime value).
      %dyn_len = arith.muli %c2_i32, %c3_i32 : i32
      aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{'aie.dma_bd' op Padding with a runtime len operand is not yet supported}}
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = %dyn_len
                   sizes = [2] strides = [128]
                   pad [<const_pad_before = 2, const_pad_after = 1>]
                   pad_value = 0)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
