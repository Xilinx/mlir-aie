// RUN: aie-opt --split-input-file --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --aie-objectfifo-lower-cores --aie-objectfifo-lower-dmas --aie-assign-lock-ids --aie-assign-buffer-addresses %s | FileCheck %s --check-prefix=LOWER
// RUN: aie-opt --split-input-file --aie-objectfifo-allocate %s -o %t
// RUN: aie-opt --split-input-file --aie-objectfifo-allocate %t -o %t2
// RUN: diff %t %t2

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The nominal pool tile is not a user. Fixed buffers can be local to a distant
// core, and generated locks must be placed where that core can reach them.
module @relocated_core_locks {
  aie.device(npu2) {
    %home = aie.tile(0, 2)
    %reader = aie.tile(2, 2)
    %b = aie.buffer(%reader) {sym_name = "b"} : memref<16xi32>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @reader(%reader) drains @p
    %core = aie.core(%reader) {
      %obj = aie.objectfifo.acquire @reader (1) : memref<16xi32>
      aie.objectfifo.release @reader (1)
      aie.end
    }
  }
}
// CHECK-LABEL: module @relocated_core_locks
// CHECK: %[[READER:.*]] = aie.tile(2, 2)
// CHECK: aie.lock(%[[READER]]) {{.*}}sym_name = "p_prod_lock_0"
// CHECK: aie.lock(%[[READER]]) {{.*}}sym_name = "p_cons_lock_0"
// LOWER-LABEL: module @relocated_core_locks
// LOWER: %[[READER:.*]] = aie.tile(2, 2)
// LOWER: %[[FREE:.*]] = aie.lock(%[[READER]], {{[0-9]+}}) {{.*}}sym_name = "p_prod_lock_0"
// LOWER: %[[FULL:.*]] = aie.lock(%[[READER]], {{[0-9]+}}) {{.*}}sym_name = "p_cons_lock_0"
// LOWER: aie.core(%[[READER]])
// LOWER: aie.use_lock(%[[FULL]], AcquireGreaterEqual,
// LOWER: aie.use_lock(%[[FREE]], Release,

// -----

// A core and a compute-tile DMA share this segment. The DMA requires local
// locks; the core can use that same lock module from its neighboring tile.
module @core_and_dma_locks {
  aie.device(npu2) {
    %home = aie.tile(0, 2)
    %memory = aie.tile(0, 3)
    %reader = aie.tile(0, 4)
    %b = aie.buffer(%memory) {sym_name = "b"} : memref<16xi32>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @writer(%memory) fills @p
    aie.objectfifo.core_endpoint @reader(%reader) drains @p
    %core = aie.core(%reader) {
      %obj = aie.objectfifo.acquire @reader (1) : memref<16xi32>
      aie.objectfifo.release @reader (1)
      aie.end
    }
  }
}
// CHECK-LABEL: module @core_and_dma_locks
// CHECK: %[[MEMORY:.*]] = aie.tile(0, 3)
// CHECK: aie.lock(%[[MEMORY]]) {{.*}}sym_name = "p_prod_lock_0"
// CHECK: aie.lock(%[[MEMORY]]) {{.*}}sym_name = "p_cons_lock_0"
// LOWER-LABEL: module @core_and_dma_locks
// LOWER: %[[MEMORY:.*]] = aie.tile(0, 3)
// LOWER: %[[FREE:.*]] = aie.lock(%[[MEMORY]], {{[0-9]+}}) {{.*}}sym_name = "p_prod_lock_0"
// LOWER: %[[FULL:.*]] = aie.lock(%[[MEMORY]], {{[0-9]+}}) {{.*}}sym_name = "p_cons_lock_0"
// LOWER: aie.core
// LOWER: aie.use_lock(%[[FULL]], AcquireGreaterEqual,
// LOWER: aie.use_lock(%[[FREE]], Release,

// -----

// A pool with delegated buffers does not constrain the lock module's tile
// kind: the concrete core user determines which generated locks are reachable.
module @delegated_core_locks {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    %reader = aie.tile(0, 2)
    %b = aie.buffer(%reader) {sym_name = "b"} : memref<16xi32>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @reader(%reader) drains @p
  }
}
// CHECK-LABEL: module @delegated_core_locks
// CHECK: %[[READER:.*]] = aie.tile(0, 2)
// CHECK: aie.lock(%[[READER]]) {{.*}}sym_name = "p_prod_lock_0"
// CHECK: aie.lock(%[[READER]]) {{.*}}sym_name = "p_cons_lock_0"
// LOWER-LABEL: module @delegated_core_locks
// LOWER: %[[READER:.*]] = aie.tile(0, 2)
// LOWER: aie.lock(%[[READER]], {{[0-9]+}}) {{.*}}sym_name = "p_prod_lock_0"
// LOWER: aie.lock(%[[READER]], {{[0-9]+}}) {{.*}}sym_name = "p_cons_lock_0"

// -----

module @relocated_binary_locks {
  aie.device(xcvc1902) {
    %home = aie.tile(0, 2)
    %reader = aie.tile(2, 2)
    %b = aie.buffer(%reader) {sym_name = "b"} : memref<16xi32>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @reader(%reader) drains @p
    %core = aie.core(%reader) {
      %obj = aie.objectfifo.acquire @reader (1) : memref<16xi32>
      aie.objectfifo.release @reader (1)
      aie.end
    }
  }
}
// CHECK-LABEL: module @relocated_binary_locks
// CHECK: %[[READER:.*]] = aie.tile(2, 2)
// CHECK: aie.lock(%[[READER]]) {{.*}}sym_name = "p_lock_0"
// LOWER-LABEL: module @relocated_binary_locks
// LOWER: %[[READER:.*]] = aie.tile(2, 2)
// LOWER: %[[LOCK:.*]] = aie.lock(%[[READER]], {{[0-9]+}}) {{.*}}sym_name = "p_lock_0"
// LOWER: aie.core
// LOWER: scf.yield %[[LOCK]]
// LOWER: aie.use_lock

// -----

// Core endpoints may access neighbor locks. Keep an already reachable nominal
// placement, rather than forcing every core's locks to be local.
module @neighbor_core_locks {
  aie.device(npu2) {
    %home = aie.tile(0, 2)
    %reader = aie.tile(0, 3)
    %b = aie.buffer(%home) {sym_name = "b"} : memref<16xi32>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @reader(%reader) drains @p
    %core = aie.core(%reader) {
      %obj = aie.objectfifo.acquire @reader (1) : memref<16xi32>
      aie.objectfifo.release @reader (1)
      aie.end
    }
  }
}
// CHECK-LABEL: module @neighbor_core_locks
// CHECK: %[[HOME:.*]] = aie.tile(0, 2)
// CHECK: aie.lock(%[[HOME]]) {{.*}}sym_name = "p_prod_lock_0"
// CHECK: aie.lock(%[[HOME]]) {{.*}}sym_name = "p_cons_lock_0"
// LOWER-LABEL: module @neighbor_core_locks
// LOWER: %[[HOME:.*]] = aie.tile(0, 2)
// LOWER: %[[FREE:.*]] = aie.lock(%[[HOME]], {{[0-9]+}}) {{.*}}sym_name = "p_prod_lock_0"
// LOWER: %[[FULL:.*]] = aie.lock(%[[HOME]], {{[0-9]+}}) {{.*}}sym_name = "p_cons_lock_0"
// LOWER: aie.core
// LOWER: aie.use_lock(%[[FULL]], AcquireGreaterEqual,
// LOWER: aie.use_lock(%[[FREE]], Release,

// -----

// Segment locks have different users. The upper core cannot reach the lower
// core's locks, but it only uses s1; relocate that pair, not the whole pool.
module @selected_segment_locks {
  aie.device(npu2) {
    %lower = aie.tile(0, 2)
    %memory = aie.tile(0, 3)
    %upper = aie.tile(0, 4)
    %b = aie.buffer(%memory) {sym_name = "b"} : memref<32xi32>
    %free = aie.lock(%lower) {sym_name = "free", init = 1 : i32}
    %full = aie.lock(%lower) {sym_name = "full", init = 0 : i32}
    aie.objectfifo.pool @p(%lower) {depth = 1 : i32, buffers = [@b]} : memref<32xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32, produceLock = @free, consumeLock = @full}
      aie.objectfifo.segment @s1 {offset = 16 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @lower(%lower) fills @p {segments = [@s0]}
    aie.objectfifo.core_endpoint @upper(%upper) drains @p {segments = [@s1]}
    %core = aie.core(%upper) {
      %obj = aie.objectfifo.acquire @upper (1) : memref<16xi32, strided<[1], offset: 16>>
      aie.objectfifo.release @upper (1)
      aie.end
    }
  }
}
// CHECK-LABEL: module @selected_segment_locks
// CHECK: %[[MEMORY:.*]] = aie.tile(0, 3)
// CHECK: aie.lock(%[[MEMORY]]) {{.*}}sym_name = "p_prod_lock_1"
// CHECK: aie.lock(%[[MEMORY]]) {{.*}}sym_name = "p_cons_lock_1"
// CHECK: produceLock = @free
// LOWER-LABEL: module @selected_segment_locks
// LOWER: %[[MEMORY:.*]] = aie.tile(0, 3)
// LOWER: %[[FREE:.*]] = aie.lock(%[[MEMORY]], {{[0-9]+}}) {{.*}}sym_name = "p_prod_lock_1"
// LOWER: %[[FULL:.*]] = aie.lock(%[[MEMORY]], {{[0-9]+}}) {{.*}}sym_name = "p_cons_lock_1"
// LOWER: aie.core
// LOWER: aie.use_lock(%[[FULL]], AcquireGreaterEqual,
// LOWER: aie.use_lock(%[[FREE]], Release,
