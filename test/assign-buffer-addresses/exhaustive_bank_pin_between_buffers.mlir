//===- exhaustive_bank_pin_between_buffers.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A mem_bank buffer that fits only between two others, which the ranked search
// cannot reach in any order. Reduced from allocator_properties.py seed 756.
//
//   npu2 core tile, 65536 bytes, four banks of 16384
//   stack        0     .. 1024
//   "low"        5760  .. 9664    (address-pinned)
//   "mid"        10560 .. 10752   (address-pinned)
//   free:        1024 .. 5760 (4736), 9664 .. 10560 (896), 10752 .. 65536
//
// Only the top run holds "small" (7040) or "big" (42881). "banked" must lie in
// bank 1 (16384 .. 32768):
//   below both, they need 16448 + 49921 = 66369 bytes: past the tile;
//   above both, or above "big", it starts past 53633: past bank 1.
// So "small", then "banked" flush against it, then "big": 10752, 17792, 17856.
//
// Ranked, "banked" goes first (it is the most constrained) and only tries the
// flush ends of bank 1's free space, 16384 and 32704. Each leaves the two big
// buffers without room.

// RUN: aie-opt --aie-assign-buffer-addresses %s | FileCheck %s

// CHECK-DAG: aie.buffer({{.*}}) {address = 10752 : i32, {{.*}}sym_name = "small"}
// CHECK-DAG: aie.buffer({{.*}}) {address = 17792 : i32, mem_bank = 1 : i32, sym_name = "banked"}
// CHECK-DAG: aie.buffer({{.*}}) {address = 17856 : i32, {{.*}}sym_name = "big"}

module @exhaustive_bank_pin_between_buffers {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %low = aie.buffer(%t) { sym_name = "low", address = 5760 : i32 } : memref<3904xi8>
    %mid = aie.buffer(%t) { sym_name = "mid", address = 10560 : i32 } : memref<192xi8>
    %small = aie.buffer(%t) { sym_name = "small" } : memref<7040xi8>
    %banked = aie.buffer(%t) { sym_name = "banked", mem_bank = 1 : i32 } : memref<64xi8>
    %big = aie.buffer(%t) { sym_name = "big" } : memref<42881xi8>
    aie.core(%t) {
      aie.end
    } { stack_size = 1024 : i32 }
  }
}
