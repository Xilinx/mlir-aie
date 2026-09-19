// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --verify-diagnostics %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Existing locks never move, even when the core can reach every buffer.
module @unreachable_fixed_locks {
  aie.device(npu2) {
    %home = aie.tile(0, 2)
    %reader = aie.tile(2, 2)
    %b = aie.buffer(%reader) {sym_name = "b"} : memref<16xi32>
    %free = aie.lock(%home) {sym_name = "free", init = 1 : i32}
    %full = aie.lock(%home) {sym_name = "full", init = 0 : i32}
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32, produceLock = @free, consumeLock = @full}
    }
    // expected-error @+1 {{cannot access pool locks}}
    aie.objectfifo.core_endpoint @reader(%reader) drains @p
  }
}

// -----

// Unknown columns do not allow a core to access locks three rows away.
module @unreachable_partial_core_locks {
  aie.device(npu2) {
    %home = aie.tile(0, 5)
    %reader = aie.logical_tile<CoreTile>(?, 2)
    %b = aie.buffer(%reader) {sym_name = "b"} : memref<16xi32>
    %free = aie.lock(%home) {sym_name = "free", init = 1 : i32}
    %full = aie.lock(%home) {sym_name = "full", init = 0 : i32}
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32, produceLock = @free, consumeLock = @full}
    }
    // expected-error @+1 {{cannot access pool locks}}
    aie.objectfifo.core_endpoint @reader(%reader) drains @p
  }
}

// -----

// AIE1 cores use the pool's binary locks, not its segment counting locks.
module @unreachable_binary_core_locks {
  aie.device(xcvc1902) {
    %home = aie.tile(0, 2)
    %reader = aie.tile(2, 2)
    %b = aie.buffer(%reader) {sym_name = "b"} : memref<16xi32>
    %lock = aie.lock(%home) {sym_name = "lock", init = 0 : i32}
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b], locks = [@lock]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error @+1 {{cannot access pool locks}}
    aie.objectfifo.core_endpoint @reader(%reader) drains @p
  }
}

// -----

// A compute DMA cannot use neighbor locks even though a core could. Partial
// coordinates cannot make distinct, already-known rows co-locate.
module @nonlocal_core_dma_locks {
  aie.device(npu2) {
    %home = aie.tile(0, 3)
    %reader = aie.logical_tile<CoreTile>(?, 2)
    %b = aie.buffer(%reader) {sym_name = "b"} : memref<16xi32>
    %free = aie.lock(%home) {sym_name = "free", init = 1 : i32}
    %full = aie.lock(%home) {sym_name = "full", init = 0 : i32}
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32, produceLock = @free, consumeLock = @full}
    }
    // expected-error @+1 {{cannot access pool locks}}
    aie.objectfifo.dma_endpoint @reader(%reader) drains @p
  }
}
