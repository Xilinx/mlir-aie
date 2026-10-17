//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-assign-runtime-sequence-bd-ids --split-input-file %s | FileCheck %s

// A runtime task after an aiex.npu.load_pdi avoids the static BDs of the device
// it loaded as well as those of the device holding the sequence. Each load
// restarts allocation, so the ids of an earlier device's chain and of tasks
// configured before the load are free again.

// CHECK-LABEL: aie.runtime_sequence @main
// CHECK: aiex.npu.load_pdi {device_ref = @a}
// CHECK: aie.dma_bd({{.*}}) {bd_id = 3 : i32}
// CHECK: aiex.npu.load_pdi {device_ref = @b}
// CHECK: aie.dma_bd({{.*}}) {bd_id = 2 : i32}
// CHECK: aiex.dma_await_task
module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<1024xi32>
    %mem = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.runtime_sequence @main() {
      aiex.npu.load_pdi {device_ref = @a}
      %t0 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t0)
      aiex.npu.load_pdi {device_ref = @b}
      %t1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t0)
    }
  }

  aie.device(npu2) @a {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<1024xi32>
    %mem = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%buf : memref<1024xi32> offset = 256 len = 256) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }

  aie.device(npu2) @b {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<1024xi32>
    %mem = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
  }
}

// -----

// A task configured before a load_pdi cannot be started after it.

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<1024xi32>
    aie.runtime_sequence @main() {
      %t0 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.npu.load_pdi {device_ref = @a}
      // expected-error@+1 {{starts a task configured before an aiex.npu.load_pdi}}
      aiex.dma_start_task(%t0)
    }
  }

  aie.device(npu2) @a {
    %tile_0_1 = aie.tile(0, 1)
  }
}

// -----

// A load_pdi that names no device hides the loaded PDI's static BDs, so a task
// after it cannot be given an id.

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<1024xi32>
    aie.runtime_sequence @main() {
      // expected-note@+1 {{the PDI is loaded here}}
      aiex.npu.load_pdi {id = 1 : i32}
      %t0 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        // expected-error@+1 {{needs a buffer descriptor ID, but the aiex.npu.load_pdi before it does not name a device in this module}}
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%t0)
    }
  }
}

// -----

// An explicit bd_id needs no allocation, so it may follow such a load.

// CHECK-LABEL: aie.runtime_sequence @main
// CHECK: aiex.npu.load_pdi {id = 1 : i32}
// CHECK: aie.dma_bd({{.*}}) {bd_id = 5 : i32}
module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<1024xi32>
    aie.runtime_sequence @main() {
      aiex.npu.load_pdi {id = 1 : i32}
      %t0 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256) {bd_id = 5 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
    }
  }
}
