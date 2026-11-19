// RUN: aie-opt --aie-objectfifo-allocate %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Buffers are placed largest buffer first. @whole's objects, the smallest
// here, make one 300032-byte buffer that only fits at home before @big's
// objects take the room; placed by object size, it comes last, and the spill
// search runs out of retries over @small's choices before it revisits @big's.
// CHECK-LABEL: module @contiguous_first
// CHECK-DAG: %[[M0:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[M1:.*]] = aie.tile(1, 1)
// CHECK-DAG: aie.buffer(%[[M0]]) {sym_name = "whole__buf0"} : memref<300032xi8>
// CHECK-DAG: aie.buffer(%[[M0]]) {sym_name = "big__buf0"}
// CHECK-DAG: aie.buffer(%[[M1]]) {sym_name = "big__buf1"}
module @contiguous_first {
  aie.device(npu2) {
    %m0 = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    %r0 = aie.buffer(%m0) : memref<124224xi8>
    %r1 = aie.buffer(%m1) : memref<324288xi8>
    aie.objectfifo.pool @big(%m0) {depth = 2 : i32} : memref<100032xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 100032 : i32}
    }
    aie.objectfifo.pool @small(%m0) {depth = 9 : i32} : memref<10240xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 10240 : i32}
    }
    aie.objectfifo.pool @whole(%m0) {contiguous, depth = 32 : i32} : memref<9376xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 9376 : i32}
    }
    aie.objectfifo.dma_endpoint @big_dma(%m0) fills @big
    aie.objectfifo.dma_endpoint @small_dma(%m0) fills @small
    aie.objectfifo.dma_endpoint @whole_dma(%m0) fills @whole
  }
}
