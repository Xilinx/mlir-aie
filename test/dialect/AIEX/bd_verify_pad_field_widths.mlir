//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Memtile MM2S pad fields have hardware bit widths (D0 6-bit/max 63, D1
// 5-bit/max 31, D2 4-bit/max 15, in 32-bit words) that were previously
// unchecked anywhere; lib/Dialect/AIEX/Transforms/AIEDmaToNpu.cpp silently
// masks with `& 0x3F` / `& 0x1F` / `& 0xF` at word-packing time instead of
// rejecting. The check lives in NpuWriteBdOp::verify (see the comment there
// for why): this exercises it via the runtime-sequence task path
// (aiex.dma_configure_task -> aiex.npu.writebd), the path with hardware
// evidence for this defect and the only path that reaches NpuWriteBdOp.
//
// RUN: aie-opt --verify-diagnostics --split-input-file --aie-dma-tasks-to-npu %s

// D0 pad_before = 64, one past the 6-bit/63 limit.
module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) { address = 0x0 : i32 } : memref<128xi32>
    aie.runtime_sequence(%arg0: memref<128xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        // expected-error@+1 {{D0 pad_before/pad_after exceeds the [0:63] range.}}
        aie.dma_bd(%buf : memref<128xi32> offset = 0 len = 4 sizes = [4] strides = [1] pad [<const_pad_before = 64, const_pad_after = 0>]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

// D0 pad_before = 63 is exactly the 6-bit field's maximum and must remain
// accepted.
module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) { address = 0x0 : i32 } : memref<128xi32>
    aie.runtime_sequence(%arg0: memref<128xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf : memref<128xi32> offset = 0 len = 4 sizes = [4] strides = [1] pad [<const_pad_before = 63, const_pad_after = 0>]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}
