//===- mmult.mlir ----------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s

// CHECK: %[[T1:.*]] = aie.tile(7, 0)
// CHECK: %[[T3:.*]] = aie.tile(8, 3)
// CHECK: %[[T15:.*]] = aie.tile(6, 0)
// CHECK: %[[T17:.*]] = aie.tile(7, 3)
// CHECK: %[[T29:.*]] = aie.tile(3, 0)
// CHECK: %[[T31:.*]] = aie.tile(8, 2)
// CHECK: %[[T43:.*]] = aie.tile(2, 0)
// CHECK: %[[T45:.*]] = aie.tile(7, 2)
//
// CHECK: aie.flow(%[[T1]], DMA : 0, %[[T3]], DMA : 0)
// CHECK: aie.flow(%[[T1]], DMA : 1, %[[T3]], DMA : 1)
// CHECK: aie.flow(%[[T3]], DMA : 0, %[[T29]], DMA : 1)
// CHECK: aie.flow(%[[T15]], DMA : 0, %[[T17]], DMA : 0)
// CHECK: aie.flow(%[[T15]], DMA : 1, %[[T17]], DMA : 1)
// CHECK: aie.flow(%[[T17]], DMA : 0, %[[T29]], DMA : 0)
// CHECK: aie.flow(%[[T29]], DMA : 0, %[[T31]], DMA : 0)
// CHECK: aie.flow(%[[T29]], DMA : 1, %[[T31]], DMA : 1)
// CHECK: aie.flow(%[[T31]], DMA : 0, %[[T43]], DMA : 1)
// CHECK: aie.flow(%[[T43]], DMA : 0, %[[T45]], DMA : 0)
// CHECK: aie.flow(%[[T43]], DMA : 1, %[[T45]], DMA : 1)
// CHECK: aie.flow(%[[T45]], DMA : 0, %[[T43]], DMA : 0)


module @aie.herd_0  {
  aie.device(xcvc1902) {
    %0 = aie.tile(7, 1)
    %1 = aie.tile(7, 0)
    %2 = aie.tile(1, 1)
    %3 = aie.tile(8, 3)
    %4 = aie.lock(%3, 1)
    %5 = aie.lock(%3, 3)
    %6 = aie.buffer(%3) {sym_name = "buf11"} : memref<16x16xf32, 2>
    %7 = aie.lock(%3, 2)
    %8 = aie.buffer(%3) {sym_name = "buf10"} : memref<16x16xf32, 2>
    %9 = aie.lock(%3, 0)
    %10 = aie.buffer(%3) {sym_name = "buf9"} : memref<16x16xf32, 2>
    %11 = aie.mem(%3)  {
      %63 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%9, Acquire, 0)
      aie.dma_bd(%10 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%9, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%4, Acquire, 0)
      aie.dma_bd(%6 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%4, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %64 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%7, Acquire, 0)
      aie.dma_bd(%8 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%7, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %65 = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%5, Acquire, 1)
      aie.dma_bd(%6 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%5, Release, 0)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      aie.end
    }
    %13 = aie.tile(6, 2)
    %14 = aie.tile(6, 1)
    %15 = aie.tile(6, 0)
    %16 = aie.tile(0, 1)
    %17 = aie.tile(7, 3)
    %18 = aie.lock(%17, 1)
    %19 = aie.lock(%17, 3)
    %20 = aie.buffer(%17) {sym_name = "buf8"} : memref<16x16xf32, 2>
    %21 = aie.lock(%17, 2)
    %22 = aie.buffer(%17) {sym_name = "buf7"} : memref<16x16xf32, 2>
    %23 = aie.lock(%17, 0)
    %24 = aie.buffer(%17) {sym_name = "buf6"} : memref<16x16xf32, 2>
    %25 = aie.mem(%17)  {
      %63 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%23, Acquire, 0)
      aie.dma_bd(%24 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%23, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%18, Acquire, 0)
      aie.dma_bd(%20 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%18, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %64 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%21, Acquire, 0)
      aie.dma_bd(%22 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%21, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %65 = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%19, Acquire, 1)
      aie.dma_bd(%20 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%19, Release, 0)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      aie.end
    }
    %27 = aie.tile(3, 2)
    %28 = aie.tile(3, 1)
    %29 = aie.tile(3, 0)
    %30 = aie.tile(1, 0)
    %31 = aie.tile(8, 2)
    %32 = aie.lock(%31, 1)
    %33 = aie.lock(%31, 3)
    %34 = aie.buffer(%31) {sym_name = "buf5"} : memref<16x16xf32, 2>
    %35 = aie.lock(%31, 2)
    %36 = aie.buffer(%31) {sym_name = "buf4"} : memref<16x16xf32, 2>
    %37 = aie.lock(%31, 0)
    %38 = aie.buffer(%31) {sym_name = "buf3"} : memref<16x16xf32, 2>
    %39 = aie.mem(%31)  {
      %63 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%37, Acquire, 0)
      aie.dma_bd(%38 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%37, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%32, Acquire, 0)
      aie.dma_bd(%34 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%32, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %64 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%35, Acquire, 0)
      aie.dma_bd(%36 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%35, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %65 = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%33, Acquire, 1)
      aie.dma_bd(%34 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%33, Release, 0)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      aie.end
    }
    %41 = aie.tile(2, 2)
    %42 = aie.tile(2, 1)
    %43 = aie.tile(2, 0)
    %44 = aie.tile(0, 0)
    %45 = aie.tile(7, 2)
    %46 = aie.lock(%45, 1)
    %47 = aie.lock(%45, 3)
    %48 = aie.buffer(%45) {sym_name = "buf2"} : memref<16x16xf32, 2>
    %49 = aie.lock(%45, 2)
    %50 = aie.buffer(%45) {sym_name = "buf1"} : memref<16x16xf32, 2>
    %51 = aie.lock(%45, 0)
    %52 = aie.buffer(%45) {sym_name = "buf0"} : memref<16x16xf32, 2>
    %53 = aie.mem(%45)  {
      %63 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%51, Acquire, 0)
      aie.dma_bd(%52 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%51, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%46, Acquire, 0)
      aie.dma_bd(%48 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%46, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %64 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%49, Acquire, 0)
      aie.dma_bd(%50 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%49, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %65 = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%47, Acquire, 1)
      aie.dma_bd(%48 : memref<16x16xf32, 2>) { len = 256 : i32 }
      aie.use_lock(%47, Release, 0)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      aie.end
    }
    %55 = aie.switchbox(%43)  {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, North : 1>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    aie.flow(%42, South : 0, %45, DMA : 0)
    aie.flow(%42, South : 1, %45, DMA : 1)
    aie.flow(%45, DMA : 0, %42, South : 0)
    %56 = aie.switchbox(%29)  {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, North : 1>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    aie.flow(%28, South : 0, %31, DMA : 0)
    aie.flow(%28, South : 1, %31, DMA : 1)
    aie.flow(%31, DMA : 0, %42, South : 1)
    %57 = aie.switchbox(%15)  {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, North : 1>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    aie.flow(%14, South : 0, %17, DMA : 0)
    aie.flow(%14, South : 1, %17, DMA : 1)
    aie.flow(%17, DMA : 0, %28, South : 0)
    %58 = aie.switchbox(%1)  {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, North : 1>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    aie.flow(%0, South : 0, %3, DMA : 0)
    aie.flow(%0, South : 1, %3, DMA : 1)
    aie.flow(%3, DMA : 0, %28, South : 1)
    %59 = aie.shim_mux(%43)  {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %60 = aie.shim_mux(%29)  {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %61 = aie.shim_mux(%15)  {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %62 = aie.shim_mux(%1)  {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
  }
}
