// RUN: aie-opt --aie-objectfifo-allocate %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Each MemTile has room for one object. @middle spills first, and both of its
// neighbors are equally empty, but only (3, 1) leaves (1, 1) free for the
// spill of @edge, whose only neighbor it is.
module @spill_backtrack {
  aie.device(npu2) {
    %m0 = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    %m2 = aie.tile(2, 1)
    %m3 = aie.tile(3, 1)
    %r0 = aie.buffer(%m0) : memref<324288xi8>
    %r1 = aie.buffer(%m1) : memref<324288xi8>
    %r2 = aie.buffer(%m2) : memref<324288xi8>
    %r3 = aie.buffer(%m3) : memref<324288xi8>
    aie.objectfifo.pool @middle(%m2) {depth = 2 : i32} : memref<200000xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 200000 : i32}
    }
    aie.objectfifo.pool @edge(%m0) {depth = 2 : i32} : memref<200000xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 200000 : i32}
    }
    aie.objectfifo.dma_endpoint @middle_dma(%m2) fills @middle
    aie.objectfifo.dma_endpoint @edge_dma(%m0) fills @edge
  }
}
// CHECK-LABEL: module @spill_backtrack
// CHECK-DAG: %[[M0:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[M1:.*]] = aie.tile(1, 1)
// CHECK-DAG: %[[M2:.*]] = aie.tile(2, 1)
// CHECK-DAG: %[[M3:.*]] = aie.tile(3, 1)
// CHECK-DAG: aie.buffer(%[[M2]]) {sym_name = "middle_buff_0"}
// CHECK-DAG: aie.buffer(%[[M3]]) {sym_name = "middle_buff_1"}
// CHECK-DAG: aie.buffer(%[[M0]]) {sym_name = "edge_buff_0"}
// CHECK-DAG: aie.buffer(%[[M1]]) {sym_name = "edge_buff_1"}
