//===- test_place_resident.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A buffer resident for a memref.global keeps the tile and address the
// first device, in structural order, gave the global's buffer: later devices
// pin it there. Both placers honour the pin; with the devices written in the
// other order and renamed, it lands alike.

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=1' %s | FileCheck %s

// CHECK-LABEL: module @resident
// CHECK: aie.buffer(%tile_3_4) {address = 1024 : i32, resident = @w, sym_name = "wa"}
// CHECK: aie.buffer(%tile_3_4) {address = 1024 : i32, resident = @w, sym_name = "wb"}
// CHECK-LABEL: module @resident_permuted
// CHECK: aie.buffer(%tile_3_4) {address = 1024 : i32, resident = @v, sym_name = "second"}
// CHECK: aie.buffer(%tile_3_4) {address = 1024 : i32, resident = @v, sym_name = "first"}

module @resident {
  memref.global "private" @w : memref<256xi32>
  aie.device(npu2) @a {
    %t = aie.logical_tile<CoreTile>(3, 4)
    %buf = aie.buffer(%t) {sym_name = "wa", address = 1024 : i32, resident = @w} : memref<256xi32>
    aie.core(%t) {
      aie.end
    }
  }
  aie.device(npu2) @b {
    %t = aie.logical_tile<CoreTile>(?, ?)
    %buf = aie.buffer(%t) {sym_name = "wb", resident = @w} : memref<256xi32>
    %u = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @f(%u, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %v = memref.load %buf[%c0] : memref<256xi32>
      aie.end
    }
  }
  aie.device(npu2) @main {
    aie.runtime_sequence @run() {
      aiex.npu.load_pdi {device_ref = @a}
      aiex.npu.load_pdi {device_ref = @b}
    }
  }
}

// -----

module @resident_permuted {
  memref.global "private" @v : memref<256xi32>
  aie.device(npu2) @driver {
    aie.runtime_sequence @go() {
      aiex.npu.load_pdi {device_ref = @first}
      aiex.npu.load_pdi {device_ref = @second}
    }
  }
  aie.device(npu2) @second {
    %x = aie.logical_tile<CoreTile>(?, ?)
    %y = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @g(%y, {%x}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %held = aie.buffer(%x) {sym_name = "second", resident = @v} : memref<256xi32>
    aie.core(%x) {
      %c0 = arith.constant 0 : index
      %v = memref.load %held[%c0] : memref<256xi32>
      aie.end
    }
  }
  aie.device(npu2) @first {
    %x = aie.logical_tile<CoreTile>(3, 4)
    %held = aie.buffer(%x) {sym_name = "first", address = 1024 : i32, resident = @v} : memref<256xi32>
    aie.core(%x) {
      aie.end
    }
  }
}

// -----

// @first ranks ahead of @second (its pad buffer), but only @second's twin has
// an address: it is pinned on @first's twin too.
// CHECK-LABEL: module @resident_later_address
// CHECK: aie.buffer(%{{.*}}) {address = 4096 : i32, resident = @u, sym_name = "early"}
// CHECK: aie.buffer(%{{.*}}) {address = 4096 : i32, resident = @u, sym_name = "late"}

module @resident_later_address {
  memref.global "private" @u : memref<16xi32>
  aie.device(npu2) @first {
    %tile = aie.tile(0, 2)
    %pad = aie.buffer(%tile) {address = 0 : i32, sym_name = "pad"} : memref<64xi32>
    %buf = aie.buffer(%tile) {sym_name = "early", resident = @u} : memref<16xi32>
  }
  aie.device(npu2) @second {
    %tile = aie.tile(0, 2)
    %buf = aie.buffer(%tile) {address = 4096 : i32, sym_name = "late", resident = @u} : memref<16xi32>
  }
}
