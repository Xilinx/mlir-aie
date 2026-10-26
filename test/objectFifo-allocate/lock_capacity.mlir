// RUN: aie-opt --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --aie-objectfifo-allocate --aie-objectfifo-lower-dmas --aie-assign-lock-ids --aie-assign-buffer-addresses %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The destination's two existing locks and 62 planned locks fill its budget.
// Localizing the cheaper pool would need two more locks. Reject that trial
// and localize the larger pool, which already has its locks at the destination.
module {
  aie.device(npu2) {
    %dst = aie.tile(0, 1)
    %neighbor = aie.tile(1, 1)
    %free = aie.lock(%dst) {sym_name = "free", init = 1 : i32}
    %full = aie.lock(%dst) {sym_name = "full", init = 0 : i32}
    aie.objectfifo.pool @resident(%dst) {depth = 1 : i32} : memref<31xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 1 : i32}
      aie.objectfifo.segment @s1 {offset = 1 : i32, size = 1 : i32}
      aie.objectfifo.segment @s2 {offset = 2 : i32, size = 1 : i32}
      aie.objectfifo.segment @s3 {offset = 3 : i32, size = 1 : i32}
      aie.objectfifo.segment @s4 {offset = 4 : i32, size = 1 : i32}
      aie.objectfifo.segment @s5 {offset = 5 : i32, size = 1 : i32}
      aie.objectfifo.segment @s6 {offset = 6 : i32, size = 1 : i32}
      aie.objectfifo.segment @s7 {offset = 7 : i32, size = 1 : i32}
      aie.objectfifo.segment @s8 {offset = 8 : i32, size = 1 : i32}
      aie.objectfifo.segment @s9 {offset = 9 : i32, size = 1 : i32}
      aie.objectfifo.segment @s10 {offset = 10 : i32, size = 1 : i32}
      aie.objectfifo.segment @s11 {offset = 11 : i32, size = 1 : i32}
      aie.objectfifo.segment @s12 {offset = 12 : i32, size = 1 : i32}
      aie.objectfifo.segment @s13 {offset = 13 : i32, size = 1 : i32}
      aie.objectfifo.segment @s14 {offset = 14 : i32, size = 1 : i32}
      aie.objectfifo.segment @s15 {offset = 15 : i32, size = 1 : i32}
      aie.objectfifo.segment @s16 {offset = 16 : i32, size = 1 : i32}
      aie.objectfifo.segment @s17 {offset = 17 : i32, size = 1 : i32}
      aie.objectfifo.segment @s18 {offset = 18 : i32, size = 1 : i32}
      aie.objectfifo.segment @s19 {offset = 19 : i32, size = 1 : i32}
      aie.objectfifo.segment @s20 {offset = 20 : i32, size = 1 : i32}
      aie.objectfifo.segment @s21 {offset = 21 : i32, size = 1 : i32}
      aie.objectfifo.segment @s22 {offset = 22 : i32, size = 1 : i32}
      aie.objectfifo.segment @s23 {offset = 23 : i32, size = 1 : i32}
      aie.objectfifo.segment @s24 {offset = 24 : i32, size = 1 : i32}
      aie.objectfifo.segment @s25 {offset = 25 : i32, size = 1 : i32}
      aie.objectfifo.segment @s26 {offset = 26 : i32, size = 1 : i32}
      aie.objectfifo.segment @s27 {offset = 27 : i32, size = 1 : i32}
      aie.objectfifo.segment @s28 {offset = 28 : i32, size = 1 : i32}
      aie.objectfifo.segment @s29 {offset = 29 : i32, size = 1 : i32}
      aie.objectfifo.segment @s30 {offset = 30 : i32, size = 1 : i32}
    }
    aie.objectfifo.pool @cheap(%neighbor) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.pool @larger(%neighbor) {depth = 1 : i32} : memref<32xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 32 : i32,
        produceLock = @free, consumeLock = @full}
    }
    aie.objectfifo.dma_endpoint @r0(%dst) drains @cheap
    aie.objectfifo.dma_endpoint @r1(%dst) drains @cheap
    aie.objectfifo.dma_endpoint @r2(%dst) drains @cheap
    aie.objectfifo.dma_endpoint @r3(%dst) drains @cheap
    aie.objectfifo.dma_endpoint @r4(%dst) drains @larger
  }
}
// CHECK-DAG: %[[DST:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[NEIGHBOR:.*]] = aie.tile(1, 1)
// CHECK-DAG: aie.buffer(%[[DST]]) {sym_name = "larger_buff_0"}
// CHECK-DAG: aie.buffer(%[[NEIGHBOR]]) {sym_name = "cheap_buff_0"}
// CHECK-DAG: aie.lock(%[[NEIGHBOR]]) {{.*}}sym_name = "cheap_prod_lock_0"
// CHECK: @r3(%[[DST]]) drains @cheap {channelIndex = 3 : i32}
// CHECK: @r4(%[[DST]]) drains @larger {channelIndex = 4 : i32}
