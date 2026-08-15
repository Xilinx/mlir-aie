// RUN: aie-opt --aie-objectfifo-lower-dmas %s | FileCheck %s
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
      depth = 2 : i32, buffers = [@b0, @b1],
      segments = [#aie.objectfifo_segment<offset = 0, size = 16,
                                          produceLock = @free, consumeLock = @full>]
    } : memref<16xi32>
    aie.objectfifo.dma_endpoint @prod_dma(%tile12) drains @prod_pool {
      channelIndex = 0 : i32
    }

    %c0 = aie.buffer(%tile33) {sym_name = "c0"} : memref<16xi32>
    %cfree = aie.lock(%tile33) {init = 1 : i32, sym_name = "cfree"}
    %cfull = aie.lock(%tile33) {init = 0 : i32, sym_name = "cfull"}

    aie.objectfifo.pool @cons_pool(%tile33) {
      depth = 1 : i32, buffers = [@c0],
      segments = [#aie.objectfifo_segment<offset = 0, size = 16,
                                          produceLock = @cfree, consumeLock = @cfull>]
    } : memref<16xi32>
    aie.objectfifo.dma_endpoint @cons_dma(%tile33) fills @cons_pool {
      channelIndex = 1 : i32
    }
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
