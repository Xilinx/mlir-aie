//===- test_sa_core_fit_order.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=QUIET --allow-empty

// Core (1, 3) overflows with two fifos of its own, @x (2 x 16 KiB) and @y
// (1 x 32 KiB), the same 32 KiB in all. (0, 3) has room for either, (1, 2)
// only for @x's two 16 KiB buffers. The SA moves the one with the larger
// objects first, so @y takes (0, 3) and @x (1, 2), with nothing over
// capacity, whichever is declared first.
// QUIET-NOT: warning
// CHECK-LABEL: module @x_first
// CHECK-DAG: aie.objectfifo.allocate @x(%tile_1_2)
// CHECK-DAG: aie.objectfifo.allocate @y(%tile_0_3)
// CHECK-LABEL: module @y_first
// CHECK-DAG: aie.objectfifo.allocate @x(%tile_1_2)
// CHECK-DAG: aie.objectfifo.allocate @y(%tile_0_3)

module @x_first {
  aie.device(npu2_4col) {
    %a = aie.logical_tile<CoreTile>(1, 3)
    %pad = aie.buffer(%a) {sym_name = "pad"} : memref<36864xi8>
    %w = aie.logical_tile<CoreTile>(0, 3)
    %wbuf = aie.buffer(%w) {sym_name = "wbuf"} : memref<24576xi8>
    %s = aie.logical_tile<CoreTile>(1, 2)
    %s1 = aie.buffer(%s) {sym_name = "s1", address = 16384 : i32} : memref<16384xi8>
    %s3 = aie.buffer(%s) {sym_name = "s3", address = 49152 : i32} : memref<16384xi8>
    %n = aie.logical_tile<CoreTile>(1, 4)
    %nbuf = aie.buffer(%n) {sym_name = "nbuf"} : memref<61440xi8>
    %e = aie.logical_tile<CoreTile>(2, 3)
    %ebuf = aie.buffer(%e) {sym_name = "ebuf"} : memref<61440xi8>
    aie.objectfifo @x(%a, {%a}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @y(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<8192xi32>>
    aie.core(%a) {
      %px = aie.objectfifo.acquire @x(Produce, 1) : memref<4096xi32>
      aie.objectfifo.release @x(Produce, 1)
      %cx = aie.objectfifo.acquire @x(Consume, 1) : memref<4096xi32>
      aie.objectfifo.release @x(Consume, 1)
      %py = aie.objectfifo.acquire @y(Produce, 1) : memref<8192xi32>
      aie.objectfifo.release @y(Produce, 1)
      %cy = aie.objectfifo.acquire @y(Consume, 1) : memref<8192xi32>
      aie.objectfifo.release @y(Consume, 1)
      aie.end
    }
  }
}

// -----

module @y_first {
  aie.device(npu2_4col) {
    %a = aie.logical_tile<CoreTile>(1, 3)
    %pad = aie.buffer(%a) {sym_name = "pad"} : memref<36864xi8>
    %w = aie.logical_tile<CoreTile>(0, 3)
    %wbuf = aie.buffer(%w) {sym_name = "wbuf"} : memref<24576xi8>
    %s = aie.logical_tile<CoreTile>(1, 2)
    %s1 = aie.buffer(%s) {sym_name = "s1", address = 16384 : i32} : memref<16384xi8>
    %s3 = aie.buffer(%s) {sym_name = "s3", address = 49152 : i32} : memref<16384xi8>
    %n = aie.logical_tile<CoreTile>(1, 4)
    %nbuf = aie.buffer(%n) {sym_name = "nbuf"} : memref<61440xi8>
    %e = aie.logical_tile<CoreTile>(2, 3)
    %ebuf = aie.buffer(%e) {sym_name = "ebuf"} : memref<61440xi8>
    aie.objectfifo @y(%a, {%a}, 1 : i32) : !aie.objectfifo<memref<8192xi32>>
    aie.objectfifo @x(%a, {%a}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.core(%a) {
      %px = aie.objectfifo.acquire @x(Produce, 1) : memref<4096xi32>
      aie.objectfifo.release @x(Produce, 1)
      %cx = aie.objectfifo.acquire @x(Consume, 1) : memref<4096xi32>
      aie.objectfifo.release @x(Consume, 1)
      %py = aie.objectfifo.acquire @y(Produce, 1) : memref<8192xi32>
      aie.objectfifo.release @y(Produce, 1)
      %cy = aie.objectfifo.acquire @y(Consume, 1) : memref<8192xi32>
      aie.objectfifo.release @y(Consume, 1)
      aie.end
    }
  }
}
