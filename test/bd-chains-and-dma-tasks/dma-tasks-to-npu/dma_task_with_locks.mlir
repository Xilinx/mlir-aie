// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

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
      %c0_i32 = arith.constant 0 : i32
      %c1024_i32 = arith.constant 1024 : i32
      %t1 = aiex.dma_configure_task(%tile_0_2, S2MM, 0) {
          %c1_ul1 = arith.constant 1 : i32
          aie.use_lock(%prod_lock, Acquire, %c1_ul1)
          aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 1024) {bd_id = 0 : i32}
          %c1_ul2 = arith.constant 1 : i32
          aie.use_lock(%cons_lock, Release, %c1_ul2)
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
// CHECK-SAME: row = 2
// CHECK-SAME: use_next_bd = 1
// CHECK-NOT: aiex.npu.write32
module @test_core_tile_looping_with_locks {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) { address = 0x0 : i32 } : memref<4096xi32>
    %prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32}
    %cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32}

    aie.runtime_sequence(%arg0: memref<4096xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c4096_i32 = arith.constant 4096 : i32
      %t1 = aiex.dma_configure_task(%tile_0_2, S2MM, 0) {
          %c1_ul3 = arith.constant 1 : i32
          aie.use_lock(%prod_lock, Acquire, %c1_ul3)
          aie.dma_bd(%buf : memref<4096xi32> offset = 0 len = 4096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
          %c1_ul4 = arith.constant 1 : i32
          aie.use_lock(%cons_lock, Release, %c1_ul4)
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
      %c0_i32 = arith.constant 0 : i32
      %c512_i32 = arith.constant 512 : i32
      %t1 = aiex.dma_configure_task(%tile_0_2, MM2S, 0) {
          aie.dma_bd(%buf : memref<512xi32> offset = 0 len = 512) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}



// -----

//===----------------------------------------------------------------------===//
// Test 4: Memtile BD with locks - verify lock IDs are offset correctly
// Memtile internal locks are accessed at base index = getNumLocks() (64),
// so logical lock ID 0 becomes physical lock ID 64.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_memtile_with_locks
// CHECK: aiex.npu.writebd
// CHECK-SAME: lock_acq_enable = 1
// CHECK-SAME: lock_acq_id = 64
// CHECK-SAME: lock_acq_val = 1
// CHECK-SAME: lock_rel_id = 65
// CHECK-SAME: lock_rel_val = 1
// CHECK-SAME: row = 1
// CHECK: aiex.npu.maskwrite32
module @test_memtile_with_locks {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) { address = 0x1000 : i32 } : memref<1024xi32>
    %prod_lock = aie.lock(%tile_0_1, 0) {init = 1 : i32}
    %cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32}

    aie.runtime_sequence(%arg0: memref<1024xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c1024_i32 = arith.constant 1024 : i32
      %t1 = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
          %c1_ul5 = arith.constant 1 : i32
          aie.use_lock(%prod_lock, Acquire, %c1_ul5)
          aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 1024) {bd_id = 0 : i32}
          %c1_ul6 = arith.constant 1 : i32
          aie.use_lock(%cons_lock, Release, %c1_ul6)
          aie.end
      }
    }
  }
}



// -----

//===----------------------------------------------------------------------===//
// Test 5: Memtile BD with looping (use_next_bd=1) and locks
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_memtile_looping_with_locks
// CHECK: aiex.npu.writebd
// CHECK-SAME: lock_acq_enable = 1
// CHECK-SAME: lock_acq_id = 64
// CHECK-SAME: lock_acq_val = 1
// CHECK-SAME: lock_rel_id = 65
// CHECK-SAME: lock_rel_val = 1
// CHECK-SAME: next_bd = 0
// CHECK-SAME: row = 1
// CHECK-SAME: use_next_bd = 1
// CHECK: aiex.npu.maskwrite32
module @test_memtile_looping_with_locks {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) { address = 0x0 : i32 } : memref<4096xi32>
    %prod_lock = aie.lock(%tile_0_1, 0) {init = 1 : i32}
    %cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32}

    aie.runtime_sequence(%arg0: memref<4096xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c4096_i32 = arith.constant 4096 : i32
      %t1 = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
          %c1_ul7 = arith.constant 1 : i32
          aie.use_lock(%prod_lock, Acquire, %c1_ul7)
          aie.dma_bd(%buf : memref<4096xi32> offset = 0 len = 4096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
          %c1_ul8 = arith.constant 1 : i32
          aie.use_lock(%cons_lock, Release, %c1_ul8)
          aie.end
      }
    }
  }
}



// -----

//===----------------------------------------------------------------------===//
// Test 6: Memtile BD without locks - verify no lock fields set
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_memtile_without_locks
// CHECK: aiex.npu.writebd
// CHECK-SAME: lock_acq_enable = 0
// CHECK-SAME: lock_acq_id = 0
// CHECK-SAME: lock_acq_val = 0
// CHECK-SAME: lock_rel_id = 0
// CHECK-SAME: lock_rel_val = 0
// CHECK-SAME: row = 1
// CHECK: aiex.npu.maskwrite32
module @test_memtile_without_locks {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) { address = 0x0 : i32 } : memref<512xi32>

    aie.runtime_sequence(%arg0: memref<512xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c512_i32 = arith.constant 512 : i32
      %t1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
          aie.dma_bd(%buf : memref<512xi32> offset = 0 len = 512) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}



// -----

//===----------------------------------------------------------------------===//
// Test 7: Memtile BD with AcquireGreaterEqual lock action
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_memtile_with_acquire_ge_lock
// CHECK: aiex.npu.writebd
// CHECK-SAME: lock_acq_enable = 1
// CHECK-SAME: lock_acq_id = 64
// CHECK-SAME: lock_acq_val = -2
// CHECK-SAME: lock_rel_id = 65
// CHECK-SAME: lock_rel_val = 1
// CHECK-SAME: row = 1
// CHECK: aiex.npu.maskwrite32
module @test_memtile_with_acquire_ge_lock {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) { address = 0x2000 : i32 } : memref<2048xi32>
    %prod_lock = aie.lock(%tile_0_1, 0) {init = 2 : i32}
    %cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32}

    aie.runtime_sequence(%arg0: memref<2048xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %t1 = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
          %c2_ul9 = arith.constant 2 : i32
          aie.use_lock(%prod_lock, AcquireGreaterEqual, %c2_ul9)
          aie.dma_bd(%buf : memref<2048xi32> offset = 0 len = 2048) {bd_id = 0 : i32}
          %c1_ul10 = arith.constant 1 : i32
          aie.use_lock(%cons_lock, Release, %c1_ul10)
          aie.end
      }
    }
  }
}



// -----

//===----------------------------------------------------------------------===//
// Test 8: Memtile BD chain with locks on multiple BDs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_memtile_chain_with_locks
// CHECK: aiex.npu.writebd
// CHECK-SAME: lock_acq_enable = 1
// CHECK-SAME: lock_acq_id = 64
// CHECK-SAME: lock_rel_id = 65
// CHECK-SAME: next_bd = 1
// CHECK-SAME: row = 1
// CHECK-SAME: use_next_bd = 1
// CHECK: aiex.npu.maskwrite32
// CHECK: aiex.npu.writebd
// CHECK-SAME: lock_acq_enable = 1
// CHECK-SAME: lock_acq_id = 64
// CHECK-SAME: lock_rel_id = 65
// CHECK-SAME: row = 1
// CHECK: aiex.npu.maskwrite32
module @test_memtile_chain_with_locks {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %buf0 = aie.buffer(%tile_0_1) { address = 0x0 : i32 } : memref<512xi32>
    %buf1 = aie.buffer(%tile_0_1) { address = 0x800 : i32 } : memref<512xi32>
    %prod_lock = aie.lock(%tile_0_1, 0) {init = 2 : i32}
    %cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32}

    aie.runtime_sequence(%arg0: memref<1024xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c512_i32 = arith.constant 512 : i32
      %t1 = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
          %c1_ul11 = arith.constant 1 : i32
          aie.use_lock(%prod_lock, Acquire, %c1_ul11)
          aie.dma_bd(%buf0 : memref<512xi32> offset = 0 len = 512) {bd_id = 0 : i32}
          %c1_ul12 = arith.constant 1 : i32
          aie.use_lock(%cons_lock, Release, %c1_ul12)
          aie.next_bd ^bd1
        ^bd1:
          %c1_ul13 = arith.constant 1 : i32
          aie.use_lock(%prod_lock, Acquire, %c1_ul13)
          aie.dma_bd(%buf1 : memref<512xi32> offset = 0 len = 512) {bd_id = 1 : i32}
          %c1_ul14 = arith.constant 1 : i32
          aie.use_lock(%cons_lock, Release, %c1_ul14)
          aie.end
      }
    }
  }
}



// -----

//===----------------------------------------------------------------------===//
// Test 9: Memtile with different lock IDs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_memtile_different_lock_ids
// CHECK: aiex.npu.writebd
// CHECK-SAME: lock_acq_enable = 1
// CHECK-SAME: lock_acq_id = 66
// CHECK-SAME: lock_acq_val = 1
// CHECK-SAME: lock_rel_id = 67
// CHECK-SAME: lock_rel_val = 1
// CHECK-SAME: row = 1
// CHECK: aiex.npu.maskwrite32
module @test_memtile_different_lock_ids {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) { address = 0x1000 : i32 } : memref<1024xi32>
    // Using lock IDs 2 and 3 instead of 0 and 1
    %prod_lock = aie.lock(%tile_0_1, 2) {init = 1 : i32}
    %cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32}

    aie.runtime_sequence(%arg0: memref<1024xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c1024_i32 = arith.constant 1024 : i32
      %t1 = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
          %c1_ul15 = arith.constant 1 : i32
          aie.use_lock(%prod_lock, Acquire, %c1_ul15)
          aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 1024) {bd_id = 0 : i32}
          %c1_ul16 = arith.constant 1 : i32
          aie.use_lock(%cons_lock, Release, %c1_ul16)
          aie.end
      }
    }
  }
}
