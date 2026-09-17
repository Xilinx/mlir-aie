// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --verify-diagnostics %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fixed buffers alone must fit, even when no pool generates new storage.
module @fixed_pool_overflow {
  aie.device(npu2) {
    // expected-error @+1 {{existing buffers require 524289 bytes, exceeding MemTile capacity of 524288 bytes}}
    %mem = aie.tile(0, 1)
    %a = aie.buffer(%mem) {sym_name = "a"} : memref<524288xi8>
    %b = aie.buffer(%mem) {sym_name = "b"} : memref<1xi8>
    aie.objectfifo.pool @p(%mem) {depth = 1 : i32, buffers = [@a]} : memref<524288xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 524288 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%mem) drains @p
  }
}

// -----

// Account for all references to a physical coordinate, not just one SSA tile.
// No pool is needed to diagnose an impossible fixed allocation.
module @aliased_fixed_overflow {
  aie.device(npu2) {
    // expected-error @+1 {{existing buffers require 524289 bytes, exceeding MemTile capacity of 524288 bytes}}
    %mem = aie.tile(0, 1)
    %alias = aie.logical_tile<MemTile>(0, 1)
    %a = aie.buffer(%mem) : memref<524288xi8>
    %b = aie.buffer(%alias) : memref<1xi8>
  }
}

// -----

// Each i1 element occupies a byte, just as in BufferOp::getAllocationSize().
// The object cannot fit in the eight remaining bytes and cannot spill.
module @sub_byte_overflow {
  aie.device(npu2_1col) {
    %mem = aie.tile(0, 1)
    %reserved = aie.buffer(%mem) : memref<524280xi8>
    // expected-error @+1 {{could not place buffers in accessible memory with available capacity}}
    aie.objectfifo.pool @p(%mem) {depth = 1 : i32} : memref<9xi1> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 9 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%mem) drains @p
  }
}
