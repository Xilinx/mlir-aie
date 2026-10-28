//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-assign-runtime-sequence-bd-ids %s

// A pinned BD id that the channel cannot reach is a diagnostic rather than a
// descriptor that is configured and then never runs. See
// memtile-channel-parity.mlir for the allocation side of the same rule.

// Odd channel, id from the low half.
module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<1024xi32>
    aie.runtime_sequence @odd_channel_low_id() {
      %t = aiex.dma_configure_task(%tile_0_1, MM2S, 1) {
        // expected-error@+1 {{cannot be submitted on channel 1}}
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

// Even channel, id from the high half.
module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<1024xi32>
    aie.runtime_sequence @even_channel_high_id() {
      %t = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
        // expected-error@+1 {{cannot be submitted on channel 0}}
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256) {bd_id = 30 : i32}
        aie.end
      }
    }
  }
}
