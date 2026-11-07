// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-bd-ids --split-input-file --verify-diagnostics %s | FileCheck %s

// A numeric ID present on both tiles does not make a remote tag local.
module @remote_tag {
  aie.device(npu2) {
    %x = aie.tile(0, 2)
    %y = aie.tile(1, 2)
    %bx = aie.buffer(%x) : memref<8xi32>
    %by = aie.buffer(%y) : memref<8xi32>
    aie.mem(%x) {
      aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%bx : memref<8xi32>) {bd_id = 0 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.mem(%y) {
      aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%by : memref<8xi32>) {bd_id = 0 : i32, bd_tag = 7 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %core = aie.core(%x) {
      // expected-error @+1 {{pushes a descriptor that this tile does not hold}}
      aie.dma_push_queue(MM2S : 0) {bd_tag = 7 : i32}
      aie.end
    }
  }
}

// -----

// A local tag still resolves to the assigned hardware ID.
// CHECK-LABEL: module @local_tag
// CHECK: aie.dma_push_queue(MM2S : 0) bd_id 3
module @local_tag {
  aie.device(npu2) {
    %x = aie.tile(1, 2)
    %bx = aie.buffer(%x) : memref<8xi32>
    aie.mem(%x) {
      aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%bx : memref<8xi32>) {bd_id = 3 : i32, bd_tag = 7 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %core = aie.core(%x) {
      aie.dma_push_queue(MM2S : 0) {bd_tag = 7 : i32}
      aie.end
    }
  }
}
