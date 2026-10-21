// RUN: aie-opt --aie-objectfifo-allocate %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Each MemTile has room for one object. @free fits at home and greedily takes
// that room, but @pinned's endpoints sit on either side of (1, 1), so (1, 1)
// is the only memory both reach. Only spilling @free, a choice its home did
// not force, leaves (1, 1) for @pinned.
module @home_backtrack {
  aie.device(npu2) {
    %m0 = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    %m2 = aie.tile(2, 1)
    %r0 = aie.buffer(%m0) : memref<324288xi8>
    %r1 = aie.buffer(%m1) : memref<324288xi8>
    %r2 = aie.buffer(%m2) : memref<324288xi8>
    aie.objectfifo.pool @free(%m1) {depth = 1 : i32} : memref<200000xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 200000 : i32}
    }
    aie.objectfifo.pool @pinned(%m1) {depth = 1 : i32} : memref<200000xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 200000 : i32}
    }
    aie.objectfifo.dma_endpoint @free_dma(%m1) fills @free
    aie.objectfifo.dma_endpoint @pinned_in(%m0) fills @pinned
    aie.objectfifo.dma_endpoint @pinned_out(%m2) drains @pinned
  }
}
// CHECK-LABEL: module @home_backtrack
// CHECK-DAG: %[[M0:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[M1:.*]] = aie.tile(1, 1)
// CHECK-DAG: aie.buffer(%[[M0]]) {sym_name = "free_buff_0"}
// CHECK-DAG: aie.buffer(%[[M1]]) {sym_name = "pinned_buff_0"}
