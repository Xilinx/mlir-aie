//===- test_place_resident_errors.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A later buffer resident for a global must fit where the first one is: the
// same tile type, no pin elsewhere, and no other address. Without an address
// on either, the data may move, which a warning says.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles %s -o /dev/null

module @resident_type {
  memref.global "private" @w : memref<256xi32>
  aie.device(npu2) @a {
    %t = aie.logical_tile<CoreTile>(3, 4)
    // expected-note@+1 {{the earlier buffer}}
    %buf = aie.buffer(%t) {sym_name = "wa", resident = @w} : memref<256xi32>
    aie.core(%t) {
      aie.end
    }
  }
  aie.device(npu2) @b {
    %m = aie.logical_tile<MemTile>(?, ?)
    // expected-error@+1 {{buffer resident for @w is on a MemTile, but the earlier buffer resident for it is on a CoreTile at (3, 4)}}
    %buf = aie.buffer(%m) {sym_name = "wb", resident = @w} : memref<256xi32>
  }
}

// -----

module @resident_pinned {
  memref.global "private" @w : memref<256xi32>
  aie.device(npu2) @a {
    %t = aie.logical_tile<CoreTile>(3, 4)
    // expected-note@+1 {{the earlier buffer}}
    %buf = aie.buffer(%t) {sym_name = "wa", resident = @w} : memref<256xi32>
    aie.core(%t) {
      aie.end
    }
  }
  aie.device(npu2) @b {
    %t = aie.logical_tile<CoreTile>(0, 2)
    // expected-error@+1 {{buffer resident for @w is on a tile pinned away from (3, 4), where the earlier buffer resident for it is}}
    %buf = aie.buffer(%t) {sym_name = "wb", resident = @w} : memref<256xi32>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = memref.load %buf[%c0] : memref<256xi32>
      aie.end
    }
  }
}

// -----

module @resident_address {
  memref.global "private" @w : memref<256xi32>
  aie.device(npu2) @a {
    %t = aie.logical_tile<CoreTile>(3, 4)
    // expected-note@+1 {{the earlier buffer}}
    %buf = aie.buffer(%t) {sym_name = "wa", address = 1024 : i32, resident = @w} : memref<256xi32>
    aie.core(%t) {
      aie.end
    }
  }
  aie.device(npu2) @b {
    %t = aie.logical_tile<CoreTile>(?, ?)
    // expected-error@+1 {{buffer resident for @w has address 2048, but the earlier buffer resident for it has 1024}}
    %buf = aie.buffer(%t) {sym_name = "wb", address = 2048 : i32, resident = @w} : memref<256xi32>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = memref.load %buf[%c0] : memref<256xi32>
      aie.end
    }
  }
}

// -----

module @resident_no_address {
  memref.global "private" @w : memref<256xi32>
  aie.device(npu2) @a {
    %t = aie.logical_tile<CoreTile>(3, 4)
    %buf = aie.buffer(%t) {sym_name = "wa", resident = @w} : memref<256xi32>
    aie.core(%t) {
      aie.end
    }
  }
  aie.device(npu2) @b {
    %t = aie.logical_tile<CoreTile>(?, ?)
    // expected-warning@+1 {{buffer resident for @w has no address, so its data may move between configurations; give it and its earlier twin one}}
    %buf = aie.buffer(%t) {sym_name = "wb", resident = @w} : memref<256xi32>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = memref.load %buf[%c0] : memref<256xi32>
      aie.end
    }
  }
}
