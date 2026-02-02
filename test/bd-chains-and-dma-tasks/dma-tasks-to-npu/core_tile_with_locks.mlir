//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 AMD Inc.

// RUN: aie-opt --aie-dma-tasks-to-npu --split-input-file %s | FileCheck %s --check-prefix=CHECK

// These tests ensure that buffer descriptor configurations on core tiles
// with lock synchronization get lowered correctly. Core tile buffers should
// NOT generate write32 operations (unlike memtile buffers).

//===----------------------------------------------------------------------===//
// Test 1: Core tile BD with locks - verify lock fields are set correctly
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_core_tile_with_locks
// CHECK: aiex.npu.writebd
// CHECK-SAME: lock_acq_enable = 1
// CHECK-SAME: lock_acq_id = 0
// CHECK-SAME: lock_acq_val = 1
// CHECK-SAME: lock_rel_id = 1
// CHECK-SAME: lock_rel_val = 1
// CHECK-SAME: row = 2
// CHECK-NOT: aiex.npu.write32
module @test_core_tile_with_locks {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) { address = 0x1000 : i32 } : memref<1024xi32>
    %prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32}
    %cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32}

    aie.runtime_sequence(%arg0: memref<1024xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_2, S2MM, 0) {
          aie.use_lock(%prod_lock, Acquire, 1)
          aie.dma_bd(%buf : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
          aie.use_lock(%cons_lock, Release, 1)
          aie.end
      }
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// Test 2: Core tile BD with looping (use_next_bd=1) and locks
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_core_tile_looping_with_locks
// CHECK: aiex.npu.writebd
// CHECK-SAME: lock_acq_enable = 1
// CHECK-SAME: next_bd = 0
// CHECK-SAME: use_next_bd = 1
// CHECK-SAME: row = 2
// CHECK-NOT: aiex.npu.write32
module @test_core_tile_looping_with_locks {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) { address = 0x0 : i32 } : memref<4096xi32>
    %prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32}
    %cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32}

    aie.runtime_sequence(%arg0: memref<4096xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_2, S2MM, 0) {
          aie.use_lock(%prod_lock, Acquire, 1)
          aie.dma_bd(%buf : memref<4096xi32>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
          aie.use_lock(%cons_lock, Release, 1)
          aie.end
      }
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// Test 3: Core tile BD without locks - verify no lock fields set
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_core_tile_without_locks
// CHECK: aiex.npu.writebd
// CHECK-SAME: lock_acq_enable = 0
// CHECK-SAME: lock_acq_id = 0
// CHECK-SAME: lock_acq_val = 0
// CHECK-SAME: lock_rel_id = 0
// CHECK-SAME: lock_rel_val = 0
// CHECK-SAME: row = 2
// CHECK-NOT: aiex.npu.write32
module @test_core_tile_without_locks {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) { address = 0x0 : i32 } : memref<512xi32>

    aie.runtime_sequence(%arg0: memref<512xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_2, MM2S, 0) {
          aie.dma_bd(%buf : memref<512xi32>, 0, 512) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}
