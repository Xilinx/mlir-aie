// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s
// RUN: aie-opt --pass-pipeline='builtin.module(aie.device(aie-objectFifo-stateful-transform,aie-assign-buffer-addresses))' %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// @deep's 30 objects need more BD ids than either channel group of the tile
// holds, so its pool is walked whole wherever its buffers go. As 30 objects
// they find no room once @x takes (0, 1), and the search runs out of retries
// before it moves @x; walked whole, @deep's one buffer goes first and @x
// spills to (1, 1). The buffers are placed again with the walk when they fit
// nowhere as split.
// CHECK-LABEL: module @walk_on_buffer_failure
// CHECK-DAG: %[[M0:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[M1:.*]] = aie.tile(1, 1)
// CHECK-DAG: aie.buffer(%[[M0]]) {sym_name = "deep__cons__buf0"} : memref<7680xi32>
// CHECK-DAG: aie.buffer(%[[M1]]) {sym_name = "x__cons__buf0"} : memref<5248xi32>
module @walk_on_buffer_failure {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %m0 = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    %fill0 = aie.buffer(%m0) {sym_name = "fill0"} : memref<493568xi8>
    %fill1 = aie.buffer(%m1) {sym_name = "fill1"} : memref<503296xi8>
    aie.objectfifo @x(%s1, {%m0}, 1 : i32) : !aie.objectfifo<memref<5248xi32>>
    aie.objectfifo @x_out(%m0, {%s1}, 1 : i32) : !aie.objectfifo<memref<5248xi32>>
    aie.objectfifo.link [@x] -> [@x_out] ([] [])
    aie.objectfifo @deep(%s0, {%m0}, 30 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @deep_out(%m0, {%s0}, 30 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@deep] -> [@deep_out] ([] [])
  }
}
