// RUN: aie-opt --aie-objectfifo-erase-pools %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pools with symbol users remain available with their buffers and locks.

module @erase {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)

    %b0 = aie.buffer(%tile12) {sym_name = "b0"} : memref<16xi32>

    aie.objectfifo.pool @spent(%tile12) {
      depth = 1 : i32, buffers = [@b0]
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }

    aie.objectfifo.pool @live(%tile12) {
      depth = 1 : i32, buffers = [@b0]
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @reader(%tile12) drains @live
  }
}

// CHECK-LABEL: @erase
// CHECK:     aie.buffer({{.*}}) {sym_name = "b0"}
// CHECK-NOT: aie.objectfifo.pool @spent
// CHECK:     aie.objectfifo.pool @live
// CHECK:     aie.objectfifo.core_endpoint @reader
