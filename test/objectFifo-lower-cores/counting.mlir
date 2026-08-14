// RUN: aie-opt --aie-objectfifo-lower-cores %s | FileCheck %s
// RUN: aie-opt --aie-objectfifo-lower-cores %s -o %t1.mlir
// RUN: aie-opt --aie-objectfifo-lower-cores %t1.mlir -o %t2.mlir
// RUN: diff %t1.mlir %t2.mlir

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An acquire names every object the core wants to hold, so it takes only the
// ones it does not hold already; a release gives back what it names and moves
// the rotating index on.

module @counting {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)

    %b0 = aie.buffer(%tile12) {sym_name = "b0"} : memref<16xi32>
    %b1 = aie.buffer(%tile12) {sym_name = "b1"} : memref<16xi32>
    %free = aie.lock(%tile12) {init = 2 : i32, sym_name = "free"}
    %full = aie.lock(%tile12) {init = 0 : i32, sym_name = "full"}

    aie.objectfifo.pool @pool(%tile12) {
      depth = 2 : i32, buffers = [@b0, @b1],
      segments = [#aie.objectfifo_segment<offset = 0, size = 16,
                                          produceLock = @free, consumeLock = @full>]
    } : memref<16xi32>
    aie.objectfifo.core_endpoint @writer(%tile12) fills @pool

    %core = aie.core(%tile12) {
      %sv = aie.objectfifo.acquire @writer (1) : !aie.objectfifosubview<memref<16xi32>>
      %e = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      aie.objectfifo.release @writer (1)
      aie.end
    }
  }
}

// CHECK-LABEL: @counting
// CHECK:   aie.core(%{{.*}}) {

// Take the shortfall between what is wanted and what is held.
// CHECK:     %[[DELTA:.*]] = arith.maxsi
// CHECK:     aie.use_lock(%free, AcquireGreaterEqual, %[[DELTA]])

// The object handed over is the one the rotating index selects.
// CHECK:     scf.index_switch
// CHECK:       scf.yield %b0
// CHECK:       scf.yield %b1

// CHECK:     aie.use_lock(%full, Release, %{{.*}})
// CHECK:     aie.end

// CHECK-NOT: aie.objectfifo.acquire
// CHECK-NOT: aie.objectfifo.release
// CHECK-NOT: aie.objectfifo.subview.access
// CHECK-NOT: aie.objectfifo.core_endpoint
