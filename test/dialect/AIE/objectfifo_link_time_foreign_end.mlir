//===- objectfifo_link_time_foreign_end.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// Every participant of a time link takes its turns on the link's tile.

aie.device(npu2) {
  %s = aie.tile(0, 0)
  %m = aie.tile(0, 1)
  %x = aie.tile(3, 4)
  %c1 = aie.tile(0, 2)
  %c2 = aie.tile(1, 2)
  aie.objectfifo @in0(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @o1(%m, {%c1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @o2(%x, {%c2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  // expected-error@+1 {{a time link's participants meet on its tile, but 'o2' is produced on another}}
  aie.objectfifo.link [@in0] -> [@o1, @o2] ([] []) {mode = #aie.link_mode<time>}
  aie.core(%x) {
    %0 = aie.objectfifo.acquire @o2(Produce, 1) : memref<16xi32>
    aie.objectfifo.release @o2(Produce, 1)
    aie.end
  }
}

// -----

aie.device(npu2) {
  %s = aie.tile(0, 0)
  %s1 = aie.tile(1, 0)
  %m = aie.tile(0, 1)
  %x = aie.tile(3, 4)
  %c = aie.tile(0, 2)
  aie.objectfifo @i1(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @i2(%s1, {%x}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  aie.objectfifo @out(%m, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  // expected-error@+1 {{a time link's participants meet on its tile, but 'i2' is consumed on others}}
  aie.objectfifo.link [@i1, @i2] -> [@out] ([] []) {mode = #aie.link_mode<time>}
  aie.core(%x) {
    %0 = aie.objectfifo.acquire @i2(Consume, 1) : memref<16xi32>
    aie.objectfifo.release @i2(Consume, 1)
    aie.end
  }
}
