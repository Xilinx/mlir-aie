// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --verify-diagnostics %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An unknown row does not make a two-column gap reachable.
module @distant_fixed_buffer {
  aie.device(npu2) {
    %home = aie.tile(2, 1)
    %reader = aie.logical_tile<MemTile>(0, ?)
    %b = aie.buffer(%home) {sym_name = "b"} : memref<16xi32>
    // expected-error @+1 {{could not place buffers in accessible memory with available capacity}}
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%reader) drains @p
  }
}

// -----

// Partial coordinates on the memory side obey the same reachability check.
module @distant_logical_buffer {
  aie.device(npu2) {
    %home = aie.logical_tile<MemTile>(2, ?)
    %writer = aie.tile(0, 1)
    %b = aie.buffer(%home) {sym_name = "b"} : memref<16xi32>
    // expected-error @+1 {{could not place buffers in accessible memory with available capacity}}
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @writer(%writer) fills @p
  }
}

// -----

// Both rows may be unknown without erasing the fixed column separation.
module @distant_generated_buffer {
  aie.device(npu2) {
    %home = aie.logical_tile<MemTile>(2, ?)
    %reader = aie.logical_tile<MemTile>(0, ?)
    // expected-error @+1 {{could not place buffers in accessible memory with available capacity}}
    aie.objectfifo.pool @p(%home) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%reader) drains @p
  }
}

// -----

// Local buffers cannot make unreachable, hand-placed locks accessible.
module @distant_fixed_locks {
  aie.device(npu2) {
    %home = aie.logical_tile<MemTile>(0, ?)
    %remote = aie.tile(2, 1)
    %b = aie.buffer(%home) {sym_name = "b"} : memref<16xi32>
    %free = aie.lock(%remote) {sym_name = "free", init = 1 : i32}
    %full = aie.lock(%remote) {sym_name = "full", init = 0 : i32}
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {
        offset = 0 : i32, size = 16 : i32,
        produceLock = @free, consumeLock = @full
      }
    }
    // expected-error @+1 {{cannot access pool locks}}
    aie.objectfifo.dma_endpoint @reader(%home) drains @p
  }
}

// -----

// Tile types also constrain unresolved positions: a core never accesses a
// MemTile's memory module, even if both coordinates are unknown.
module @incompatible_tile_types {
  aie.device(npu2) {
    %mem = aie.logical_tile<MemTile>(?, ?)
    %core = aie.logical_tile<CoreTile>(?, ?)
    // expected-error @+1 {{could not place buffers in accessible memory with available capacity}}
    aie.objectfifo.pool @p(%mem) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%core) drains @p
  }
}

// -----

// Core memory sharing is directional and row-sensitive; unknown columns
// cannot bridge a gap of three rows.
module @distant_core_rows {
  aie.device(npu2) {
    %home = aie.tile(0, 5)
    %reader = aie.logical_tile<CoreTile>(?, 2)
    %b = aie.buffer(%home) {sym_name = "b"} : memref<16xi32>
    // expected-error @+1 {{could not place buffers in accessible memory with available capacity}}
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%reader) drains @p
  }
}
