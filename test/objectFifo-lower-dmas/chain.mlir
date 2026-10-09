// RUN: aie-opt --aie-objectfifo-lower-dmas %s | FileCheck %s
// RUN: aie-opt --aie-objectfifo-lower-dmas --mlir-print-debuginfo %s | FileCheck %s --check-prefix=LOC --implicit-check-not='loc(unknown)'
// RUN: aie-opt --aie-objectfifo-lower-dmas %s -o %t1.mlir
// RUN: aie-opt --aie-objectfifo-lower-dmas %t1.mlir -o %t2.mlir
// RUN: diff %t1.mlir %t2.mlir

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// One buffer descriptor per object: the draining end waits on full objects and
// gives back free ones, the filling end the other way round.

module @chain {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    %b0 = aie.buffer(%tile12) {sym_name = "b0"} : memref<16xi32>
    %b1 = aie.buffer(%tile12) {sym_name = "b1"} : memref<16xi32>
    %free = aie.lock(%tile12) {init = 2 : i32, sym_name = "free"}
    %full = aie.lock(%tile12) {init = 0 : i32, sym_name = "full"}

    aie.objectfifo.pool @prod_pool(%tile12) {
      depth = 2 : i32, buffers = [@b0, @b1]
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {consumeLock = @full, offset = 0 : i32, produceLock = @free, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @prod_dma(%tile12) drains @prod_pool {
      channelIndex = 0 : i32
    } loc("fifo_user.py":30:2)

    %c0 = aie.buffer(%tile33) {sym_name = "c0"} : memref<16xi32>
    %cfree = aie.lock(%tile33) {init = 1 : i32, sym_name = "cfree"}
    %cfull = aie.lock(%tile33) {init = 0 : i32, sym_name = "cfull"}

    aie.objectfifo.pool @cons_pool(%tile33) {
      depth = 1 : i32, buffers = [@c0]
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {consumeLock = @cfull, offset = 0 : i32, produceLock = @cfree, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @cons_dma(%tile33) fills @cons_pool {
      channelIndex = 1 : i32
    } loc("fifo_user.py":40:2)
  }
}

// CHECK-LABEL: @chain
// CHECK:   aie.mem(%{{.*}}) {
// CHECK:     aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:   ^bb1:
// CHECK:     aie.use_lock(%full, AcquireGreaterEqual, %{{.*}})
// CHECK:     aie.dma_bd(%b0 : memref<16xi32> offset = 0 len = 16)
// CHECK:     aie.use_lock(%free, Release, %{{.*}})
// CHECK:     aie.next_bd ^bb2
// CHECK:   ^bb2:
// CHECK:     aie.use_lock(%full, AcquireGreaterEqual, %{{.*}})
// CHECK:     aie.dma_bd(%b1 : memref<16xi32> offset = 0 len = 16)
// CHECK:     aie.use_lock(%free, Release, %{{.*}})
// CHECK:     aie.next_bd ^bb1

// CHECK:   aie.mem(%{{.*}}) {
// CHECK:     aie.dma_start(S2MM, 1, ^bb1, ^bb2)
// CHECK:     aie.use_lock(%cfree, AcquireGreaterEqual, %{{.*}})
// CHECK:     aie.dma_bd(%c0 : memref<16xi32> offset = 0 len = 16)
// CHECK:     aie.use_lock(%cfull, Release, %{{.*}})

// CHECK-NOT: aie.objectfifo.dma_endpoint

// Distinct endpoints must retain distinct origins throughout their BD chains.
// LOC-DAG: aie.dma_start(MM2S, {{.*}}) loc(#[[PROD:loc[0-9]*]])
// LOC-DAG: aie.use_lock(%full, AcquireGreaterEqual, {{.*}}) loc(#[[PROD]])
// LOC-DAG: aie.dma_bd(%b0 : {{.*}}) loc(#[[PROD]])
// LOC-DAG: aie.dma_bd(%b1 : {{.*}}) loc(#[[PROD]])
// LOC-DAG: aie.next_bd {{.*}} loc(#[[PROD]])
// LOC-DAG: aie.dma_start(S2MM, {{.*}}) loc(#[[CONS:loc[0-9]*]])
// LOC-DAG: aie.dma_bd(%c0 : {{.*}}) loc(#[[CONS]])
// LOC-DAG: #[[PROD]] = loc("fifo_user.py":30:2)
// LOC-DAG: #[[CONS]] = loc("fifo_user.py":40:2)
