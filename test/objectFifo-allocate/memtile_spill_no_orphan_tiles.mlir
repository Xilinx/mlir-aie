// RUN: aie-opt --aie-objectfifo-allocate --split-input-file %s | FileCheck %s --implicit-check-not="aie.tile(2, 1)"

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// @spill's home is full, so it weighs both neighbors, which the design never
// declared. It lands on (0, 1); (2, 1) is left with nothing on it and must not
// stay behind as an empty tile.
module @spill_no_orphan_tiles {
  aie.device(npu2) {
    %m1 = aie.tile(1, 1)
    %r1 = aie.buffer(%m1) : memref<524288xi8>
    aie.objectfifo.pool @spill(%m1) {depth = 1 : i32} : memref<1024xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 1024 : i32}
    }
    aie.objectfifo.dma_endpoint @spill_dma(%m1) fills @spill
  }
}
// CHECK-LABEL: module @spill_no_orphan_tiles
// CHECK-DAG: %[[M0:.*]] = aie.tile(0, 1)
// CHECK-DAG: aie.buffer(%[[M0]]) {sym_name = "spill__buf0"}

// -----

// The tiles @a weighed and left empty are gone before @b is allocated, so
// @b's run must not look at them again.
module @spill_then_another_device {
  aie.device(npu2) @a {
    %m1 = aie.tile(1, 1)
    %r1 = aie.buffer(%m1) : memref<524288xi8>
    aie.objectfifo.pool @spill(%m1) {depth = 1 : i32} : memref<1024xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 1024 : i32}
    }
    aie.objectfifo.dma_endpoint @spill_dma(%m1) fills @spill
  }
  aie.device(npu2) @b {
    %t = aie.tile(0, 2)
  }
}
// CHECK-LABEL: module @spill_then_another_device
// CHECK: aie.device(npu2) @a
// CHECK: aie.buffer({{.*}}) {sym_name = "spill__buf0"}
// CHECK: aie.device(npu2) @b
// CHECK: aie.tile(0, 2)
