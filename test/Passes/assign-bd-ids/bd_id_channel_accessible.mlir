//===- bd_id_channel_accessible.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --verify-diagnostics --split-input-file %s

// A memtile partitions its BDs by channel parity: even channels reach bd_id
// 0-23, odd channels reach bd_id 24-47. A user-pinned bd_id on the wrong
// parity is rejected.

// dma_start: even channel with a bd_id from the odd range.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<1xi32>
    aie.memtile_dma(%t) {
        aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{assigned bd_id 24 is not accessible from channel 0 on this tile}}
        aie.dma_bd(%b : memref<1xi32>) { bd_id = 24 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

// dma: odd channel with a bd_id from the even range.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b = aie.buffer(%t) : memref<1xi32>
    %lock = aie.lock(%t) {init = 1 : i32}
    %lock2 = aie.lock(%t) {init = 0 : i32}
    aie.memtile_dma(%t) {
      %0 = aie.dma(S2MM, 1) [{
        %c1 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1)
        // expected-error@+1 {{assigned bd_id 0 is not accessible from channel 1 on this tile}}
        aie.dma_bd(%b : memref<1xi32>) { bd_id = 0 : i32 }
        %c1_1 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_1)
      }]
      aie.end
    }
  }
}
