//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-dma-tasks-to-npu --aie-dma-to-npu %s

// buffer_length is 17 bits on a mem tile and 14 on a core tile, against the
// full 32-bit word on a shim NOC tile. A length that does not fit used to be
// masked by the packer and silently transfer the wrong amount -- 200000
// granules became 68928 -- because NpuWriteBdOp::verify bounded every other BD
// field against the target model but not this one.

// Mem tile: 2^17 - 1 granules.
module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<200000xi32>
    aie.runtime_sequence @memtile_overflow() {
      // expected-error@+2 {{Buffer length exceeds the [0:131071] range}}
      %t = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf : memref<200000xi32> offset = 0 len = 200000) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

// Core tile: 2^14 - 1 granules.
module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) {address = 0 : i32} : memref<20000xi32>
    aie.runtime_sequence @coretile_overflow() {
      // expected-error@+2 {{Buffer length exceeds the [0:16383] range}}
      %t = aiex.dma_configure_task(%tile_0_2, MM2S, 0) {
        aie.dma_bd(%buf : memref<20000xi32> offset = 0 len = 20000) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}
