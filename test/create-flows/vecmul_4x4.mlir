//===- vecmul_4x4.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s
// CHECK: %[[T2:.*]] = aie.tile(47, 0)
// CHECK: %[[T4:.*]] = aie.tile(10, 5)
// CHECK: %[[T15:.*]] = aie.tile(46, 0)
// CHECK: %[[T17:.*]] = aie.tile(9, 5)
// CHECK: %[[T28:.*]] = aie.tile(43, 0)
// CHECK: %[[T30:.*]] = aie.tile(8, 5)
// CHECK: %[[T41:.*]] = aie.tile(42, 0)
// CHECK: %[[T43:.*]] = aie.tile(7, 5)
// CHECK: %[[T54:.*]] = aie.tile(35, 0)
// CHECK: %[[T55:.*]] = aie.tile(10, 4)
// CHECK: %[[T66:.*]] = aie.tile(34, 0)
// CHECK: %[[T67:.*]] = aie.tile(9, 4)
// CHECK: %[[T78:.*]] = aie.tile(27, 0)
// CHECK: %[[T80:.*]] = aie.tile(8, 4)
// CHECK: %[[T91:.*]] = aie.tile(26, 0)
// CHECK: %[[T93:.*]] = aie.tile(7, 4)
// CHECK: %[[T104:.*]] = aie.tile(19, 0)
// CHECK: %[[T105:.*]] = aie.tile(10, 3)
// CHECK: %[[T116:.*]] = aie.tile(18, 0)
// CHECK: %[[T117:.*]] = aie.tile(9, 3)
// CHECK: %[[T128:.*]] = aie.tile(11, 0)
// CHECK: %[[T130:.*]] = aie.tile(8, 3)
// CHECK: %[[T140:.*]] = aie.tile(10, 0)
// CHECK: %[[T142:.*]] = aie.tile(7, 3)
// CHECK: %[[T152:.*]] = aie.tile(7, 0)
// CHECK: %[[T153:.*]] = aie.tile(10, 2)
// CHECK: %[[T164:.*]] = aie.tile(6, 0)
// CHECK: %[[T165:.*]] = aie.tile(9, 2)
// CHECK: %[[T176:.*]] = aie.tile(3, 0)
// CHECK: %[[T178:.*]] = aie.tile(8, 2)
// CHECK: %[[T189:.*]] = aie.tile(2, 0)
// CHECK: %[[T191:.*]] = aie.tile(7, 2)

//
// CHECK: aie.flow(%[[T2]], DMA : 0, %[[T4]], DMA : 0)
// CHECK: aie.flow(%[[T2]], DMA : 1, %[[T4]], DMA : 1)
// CHECK: aie.flow(%[[T4]], DMA : 0, %[[T104]], DMA : 1)
// CHECK: aie.flow(%[[T15]], DMA : 0, %[[T17]], DMA : 0)
// CHECK: aie.flow(%[[T15]], DMA : 1, %[[T17]], DMA : 1)
// CHECK: aie.flow(%[[T17]], DMA : 0, %[[T104]], DMA : 0)
// CHECK: aie.flow(%[[T28]], DMA : 0, %[[T30]], DMA : 0)
// CHECK: aie.flow(%[[T28]], DMA : 1, %[[T30]], DMA : 1)
// CHECK: aie.flow(%[[T30]], DMA : 0, %[[T116]], DMA : 1)
// CHECK: aie.flow(%[[T41]], DMA : 0, %[[T43]], DMA : 0)
// CHECK: aie.flow(%[[T41]], DMA : 1, %[[T43]], DMA : 1)
// CHECK: aie.flow(%[[T43]], DMA : 0, %[[T116]], DMA : 0)
// CHECK: aie.flow(%[[T54]], DMA : 0, %[[T55]], DMA : 0)
// CHECK: aie.flow(%[[T54]], DMA : 1, %[[T55]], DMA : 1)
// CHECK: aie.flow(%[[T55]], DMA : 0, %[[T128]], DMA : 1)
// CHECK: aie.flow(%[[T66]], DMA : 0, %[[T67]], DMA : 0)
// CHECK: aie.flow(%[[T66]], DMA : 1, %[[T67]], DMA : 1)
// CHECK: aie.flow(%[[T67]], DMA : 0, %[[T128]], DMA : 0)
// CHECK: aie.flow(%[[T78]], DMA : 0, %[[T80]], DMA : 0)
// CHECK: aie.flow(%[[T78]], DMA : 1, %[[T80]], DMA : 1)
// CHECK: aie.flow(%[[T80]], DMA : 0, %[[T140]], DMA : 1)
// CHECK: aie.flow(%[[T91]], DMA : 0, %[[T93]], DMA : 0)
// CHECK: aie.flow(%[[T91]], DMA : 1, %[[T93]], DMA : 1)
// CHECK: aie.flow(%[[T93]], DMA : 0, %[[T140]], DMA : 0)
// CHECK: aie.flow(%[[T104]], DMA : 0, %[[T105]], DMA : 0)
// CHECK: aie.flow(%[[T104]], DMA : 1, %[[T105]], DMA : 1)
// CHECK: aie.flow(%[[T105]], DMA : 0, %[[T152]], DMA : 1)
// CHECK: aie.flow(%[[T116]], DMA : 0, %[[T117]], DMA : 0)
// CHECK: aie.flow(%[[T116]], DMA : 1, %[[T117]], DMA : 1)
// CHECK: aie.flow(%[[T117]], DMA : 0, %[[T152]], DMA : 0)
// CHECK: aie.flow(%[[T128]], DMA : 0, %[[T130]], DMA : 0)
// CHECK: aie.flow(%[[T128]], DMA : 1, %[[T130]], DMA : 1)
// CHECK: aie.flow(%[[T130]], DMA : 0, %[[T164]], DMA : 1)
// CHECK: aie.flow(%[[T140]], DMA : 0, %[[T142]], DMA : 0)
// CHECK: aie.flow(%[[T140]], DMA : 1, %[[T142]], DMA : 1)
// CHECK: aie.flow(%[[T142]], DMA : 0, %[[T164]], DMA : 0)
// CHECK: aie.flow(%[[T152]], DMA : 0, %[[T153]], DMA : 0)
// CHECK: aie.flow(%[[T152]], DMA : 1, %[[T153]], DMA : 1)
// CHECK: aie.flow(%[[T153]], DMA : 0, %[[T176]], DMA : 1)
// CHECK: aie.flow(%[[T164]], DMA : 0, %[[T165]], DMA : 0)
// CHECK: aie.flow(%[[T164]], DMA : 1, %[[T165]], DMA : 1)
// CHECK: aie.flow(%[[T165]], DMA : 0, %[[T176]], DMA : 0)
// CHECK: aie.flow(%[[T176]], DMA : 0, %[[T178]], DMA : 0)
// CHECK: aie.flow(%[[T176]], DMA : 1, %[[T178]], DMA : 1)
// CHECK: aie.flow(%[[T178]], DMA : 0, %[[T189]], DMA : 1)
// CHECK: aie.flow(%[[T189]], DMA : 0, %[[T191]], DMA : 0)
// CHECK: aie.flow(%[[T189]], DMA : 1, %[[T191]], DMA : 1)
// CHECK: aie.flow(%[[T191]], DMA : 0, %[[T189]], DMA : 0)

module @vecmul_4x4  {
  aie.device(xcvc1902) {
    %0 = aie.tile(47, 2)
    %1 = aie.tile(47, 1)
    %2 = aie.tile(47, 0)
    %3 = aie.tile(3, 3)
    %4 = aie.tile(10, 5)
    %5 = aie.lock(%4, 2)
    %6 = aie.buffer(%4) {sym_name = "buf47"} : memref<64xi32, 2>
    %7 = aie.lock(%4, 1)
    %8 = aie.buffer(%4) {sym_name = "buf46"} : memref<64xi32, 2>
    %9 = aie.lock(%4, 0)
    %10 = aie.buffer(%4) {sym_name = "buf45"} : memref<64xi32, 2>
    %11 = aie.mem(%4)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%9, Acquire, 0)
      aie.dma_bd(%10 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%9, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%7, Acquire, 0)
      aie.dma_bd(%8 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%7, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%5, Acquire, 1)
      aie.dma_bd(%6 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%5, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %12 = aie.core(%4)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%9, Acquire, 1)
      aie.use_lock(%7, Acquire, 1)
      aie.use_lock(%5, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %10[%arg0] : memref<64xi32, 2>
        %201 = affine.load %8[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %6[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%5, Release, 1)
      aie.use_lock(%7, Release, 0)
      aie.use_lock(%9, Release, 0)
      cf.br ^bb1
    }
    %13 = aie.tile(46, 2)
    %14 = aie.tile(46, 1)
    %15 = aie.tile(46, 0)
    %16 = aie.tile(2, 3)
    %17 = aie.tile(9, 5)
    %18 = aie.lock(%17, 2)
    %19 = aie.buffer(%17) {sym_name = "buf44"} : memref<64xi32, 2>
    %20 = aie.lock(%17, 1)
    %21 = aie.buffer(%17) {sym_name = "buf43"} : memref<64xi32, 2>
    %22 = aie.lock(%17, 0)
    %23 = aie.buffer(%17) {sym_name = "buf42"} : memref<64xi32, 2>
    %24 = aie.mem(%17)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%22, Acquire, 0)
      aie.dma_bd(%23 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%22, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%20, Acquire, 0)
      aie.dma_bd(%21 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%20, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%18, Acquire, 1)
      aie.dma_bd(%19 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%18, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %25 = aie.core(%17)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%22, Acquire, 1)
      aie.use_lock(%20, Acquire, 1)
      aie.use_lock(%18, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %23[%arg0] : memref<64xi32, 2>
        %201 = affine.load %21[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %19[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%18, Release, 1)
      aie.use_lock(%20, Release, 0)
      aie.use_lock(%22, Release, 0)
      cf.br ^bb1
    }
    %26 = aie.tile(43, 2)
    %27 = aie.tile(43, 1)
    %28 = aie.tile(43, 0)
    %29 = aie.tile(1, 3)
    %30 = aie.tile(8, 5)
    %31 = aie.lock(%30, 2)
    %32 = aie.buffer(%30) {sym_name = "buf41"} : memref<64xi32, 2>
    %33 = aie.lock(%30, 1)
    %34 = aie.buffer(%30) {sym_name = "buf40"} : memref<64xi32, 2>
    %35 = aie.lock(%30, 0)
    %36 = aie.buffer(%30) {sym_name = "buf39"} : memref<64xi32, 2>
    %37 = aie.mem(%30)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%35, Acquire, 0)
      aie.dma_bd(%36 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%35, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%33, Acquire, 0)
      aie.dma_bd(%34 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%33, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%31, Acquire, 1)
      aie.dma_bd(%32 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%31, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %38 = aie.core(%30)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%35, Acquire, 1)
      aie.use_lock(%33, Acquire, 1)
      aie.use_lock(%31, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %36[%arg0] : memref<64xi32, 2>
        %201 = affine.load %34[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %32[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%31, Release, 1)
      aie.use_lock(%33, Release, 0)
      aie.use_lock(%35, Release, 0)
      cf.br ^bb1
    }
    %39 = aie.tile(42, 2)
    %40 = aie.tile(42, 1)
    %41 = aie.tile(42, 0)
    %42 = aie.tile(0, 3)
    %43 = aie.tile(7, 5)
    %44 = aie.lock(%43, 2)
    %45 = aie.buffer(%43) {sym_name = "buf38"} : memref<64xi32, 2>
    %46 = aie.lock(%43, 1)
    %47 = aie.buffer(%43) {sym_name = "buf37"} : memref<64xi32, 2>
    %48 = aie.lock(%43, 0)
    %49 = aie.buffer(%43) {sym_name = "buf36"} : memref<64xi32, 2>
    %50 = aie.mem(%43)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%48, Acquire, 0)
      aie.dma_bd(%49 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%48, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%46, Acquire, 0)
      aie.dma_bd(%47 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%46, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%44, Acquire, 1)
      aie.dma_bd(%45 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%44, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %51 = aie.core(%43)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%48, Acquire, 1)
      aie.use_lock(%46, Acquire, 1)
      aie.use_lock(%44, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %49[%arg0] : memref<64xi32, 2>
        %201 = affine.load %47[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %45[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%44, Release, 1)
      aie.use_lock(%46, Release, 0)
      aie.use_lock(%48, Release, 0)
      cf.br ^bb1
    }
    %52 = aie.tile(35, 2)
    %53 = aie.tile(35, 1)
    %54 = aie.tile(35, 0)
    %55 = aie.tile(10, 4)
    %56 = aie.lock(%55, 2)
    %57 = aie.buffer(%55) {sym_name = "buf35"} : memref<64xi32, 2>
    %58 = aie.lock(%55, 1)
    %59 = aie.buffer(%55) {sym_name = "buf34"} : memref<64xi32, 2>
    %60 = aie.lock(%55, 0)
    %61 = aie.buffer(%55) {sym_name = "buf33"} : memref<64xi32, 2>
    %62 = aie.mem(%55)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%60, Acquire, 0)
      aie.dma_bd(%61 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%60, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%58, Acquire, 0)
      aie.dma_bd(%59 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%58, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%56, Acquire, 1)
      aie.dma_bd(%57 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%56, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %63 = aie.core(%55)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%60, Acquire, 1)
      aie.use_lock(%58, Acquire, 1)
      aie.use_lock(%56, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %61[%arg0] : memref<64xi32, 2>
        %201 = affine.load %59[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %57[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%56, Release, 1)
      aie.use_lock(%58, Release, 0)
      aie.use_lock(%60, Release, 0)
      cf.br ^bb1
    }
    %64 = aie.tile(34, 2)
    %65 = aie.tile(34, 1)
    %66 = aie.tile(34, 0)
    %67 = aie.tile(9, 4)
    %68 = aie.lock(%67, 2)
    %69 = aie.buffer(%67) {sym_name = "buf32"} : memref<64xi32, 2>
    %70 = aie.lock(%67, 1)
    %71 = aie.buffer(%67) {sym_name = "buf31"} : memref<64xi32, 2>
    %72 = aie.lock(%67, 0)
    %73 = aie.buffer(%67) {sym_name = "buf30"} : memref<64xi32, 2>
    %74 = aie.mem(%67)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%72, Acquire, 0)
      aie.dma_bd(%73 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%72, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%70, Acquire, 0)
      aie.dma_bd(%71 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%70, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%68, Acquire, 1)
      aie.dma_bd(%69 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%68, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %75 = aie.core(%67)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%72, Acquire, 1)
      aie.use_lock(%70, Acquire, 1)
      aie.use_lock(%68, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %73[%arg0] : memref<64xi32, 2>
        %201 = affine.load %71[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %69[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%68, Release, 1)
      aie.use_lock(%70, Release, 0)
      aie.use_lock(%72, Release, 0)
      cf.br ^bb1
    }
    %76 = aie.tile(27, 2)
    %77 = aie.tile(27, 1)
    %78 = aie.tile(27, 0)
    %79 = aie.tile(1, 2)
    %80 = aie.tile(8, 4)
    %81 = aie.lock(%80, 2)
    %82 = aie.buffer(%80) {sym_name = "buf29"} : memref<64xi32, 2>
    %83 = aie.lock(%80, 1)
    %84 = aie.buffer(%80) {sym_name = "buf28"} : memref<64xi32, 2>
    %85 = aie.lock(%80, 0)
    %86 = aie.buffer(%80) {sym_name = "buf27"} : memref<64xi32, 2>
    %87 = aie.mem(%80)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%85, Acquire, 0)
      aie.dma_bd(%86 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%85, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%83, Acquire, 0)
      aie.dma_bd(%84 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%83, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%81, Acquire, 1)
      aie.dma_bd(%82 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%81, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %88 = aie.core(%80)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%85, Acquire, 1)
      aie.use_lock(%83, Acquire, 1)
      aie.use_lock(%81, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %86[%arg0] : memref<64xi32, 2>
        %201 = affine.load %84[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %82[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%81, Release, 1)
      aie.use_lock(%83, Release, 0)
      aie.use_lock(%85, Release, 0)
      cf.br ^bb1
    }
    %89 = aie.tile(26, 2)
    %90 = aie.tile(26, 1)
    %91 = aie.tile(26, 0)
    %92 = aie.tile(0, 2)
    %93 = aie.tile(7, 4)
    %94 = aie.lock(%93, 2)
    %95 = aie.buffer(%93) {sym_name = "buf26"} : memref<64xi32, 2>
    %96 = aie.lock(%93, 1)
    %97 = aie.buffer(%93) {sym_name = "buf25"} : memref<64xi32, 2>
    %98 = aie.lock(%93, 0)
    %99 = aie.buffer(%93) {sym_name = "buf24"} : memref<64xi32, 2>
    %100 = aie.mem(%93)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%98, Acquire, 0)
      aie.dma_bd(%99 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%98, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%96, Acquire, 0)
      aie.dma_bd(%97 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%96, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%94, Acquire, 1)
      aie.dma_bd(%95 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%94, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %101 = aie.core(%93)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%98, Acquire, 1)
      aie.use_lock(%96, Acquire, 1)
      aie.use_lock(%94, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %99[%arg0] : memref<64xi32, 2>
        %201 = affine.load %97[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %95[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%94, Release, 1)
      aie.use_lock(%96, Release, 0)
      aie.use_lock(%98, Release, 0)
      cf.br ^bb1
    }
    %102 = aie.tile(19, 2)
    %103 = aie.tile(19, 1)
    %104 = aie.tile(19, 0)
    %105 = aie.tile(10, 3)
    %106 = aie.lock(%105, 2)
    %107 = aie.buffer(%105) {sym_name = "buf23"} : memref<64xi32, 2>
    %108 = aie.lock(%105, 1)
    %109 = aie.buffer(%105) {sym_name = "buf22"} : memref<64xi32, 2>
    %110 = aie.lock(%105, 0)
    %111 = aie.buffer(%105) {sym_name = "buf21"} : memref<64xi32, 2>
    %112 = aie.mem(%105)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%110, Acquire, 0)
      aie.dma_bd(%111 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%110, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%108, Acquire, 0)
      aie.dma_bd(%109 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%108, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%106, Acquire, 1)
      aie.dma_bd(%107 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%106, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %113 = aie.core(%105)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%110, Acquire, 1)
      aie.use_lock(%108, Acquire, 1)
      aie.use_lock(%106, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %111[%arg0] : memref<64xi32, 2>
        %201 = affine.load %109[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %107[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%106, Release, 1)
      aie.use_lock(%108, Release, 0)
      aie.use_lock(%110, Release, 0)
      cf.br ^bb1
    }
    %114 = aie.tile(18, 2)
    %115 = aie.tile(18, 1)
    %116 = aie.tile(18, 0)
    %117 = aie.tile(9, 3)
    %118 = aie.lock(%117, 2)
    %119 = aie.buffer(%117) {sym_name = "buf20"} : memref<64xi32, 2>
    %120 = aie.lock(%117, 1)
    %121 = aie.buffer(%117) {sym_name = "buf19"} : memref<64xi32, 2>
    %122 = aie.lock(%117, 0)
    %123 = aie.buffer(%117) {sym_name = "buf18"} : memref<64xi32, 2>
    %124 = aie.mem(%117)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%122, Acquire, 0)
      aie.dma_bd(%123 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%122, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%120, Acquire, 0)
      aie.dma_bd(%121 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%120, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%118, Acquire, 1)
      aie.dma_bd(%119 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%118, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %125 = aie.core(%117)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%122, Acquire, 1)
      aie.use_lock(%120, Acquire, 1)
      aie.use_lock(%118, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %123[%arg0] : memref<64xi32, 2>
        %201 = affine.load %121[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %119[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%118, Release, 1)
      aie.use_lock(%120, Release, 0)
      aie.use_lock(%122, Release, 0)
      cf.br ^bb1
    }
    %126 = aie.tile(11, 2)
    %127 = aie.tile(11, 1)
    %128 = aie.tile(11, 0)
    %129 = aie.tile(1, 1)
    %130 = aie.tile(8, 3)
    %131 = aie.lock(%130, 2)
    %132 = aie.buffer(%130) {sym_name = "buf17"} : memref<64xi32, 2>
    %133 = aie.lock(%130, 1)
    %134 = aie.buffer(%130) {sym_name = "buf16"} : memref<64xi32, 2>
    %135 = aie.lock(%130, 0)
    %136 = aie.buffer(%130) {sym_name = "buf15"} : memref<64xi32, 2>
    %137 = aie.mem(%130)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%135, Acquire, 0)
      aie.dma_bd(%136 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%135, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%133, Acquire, 0)
      aie.dma_bd(%134 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%133, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%131, Acquire, 1)
      aie.dma_bd(%132 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%131, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %138 = aie.core(%130)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%135, Acquire, 1)
      aie.use_lock(%133, Acquire, 1)
      aie.use_lock(%131, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %136[%arg0] : memref<64xi32, 2>
        %201 = affine.load %134[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %132[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%131, Release, 1)
      aie.use_lock(%133, Release, 0)
      aie.use_lock(%135, Release, 0)
      cf.br ^bb1
    }
    %139 = aie.tile(10, 1)
    %140 = aie.tile(10, 0)
    %141 = aie.tile(0, 1)
    %142 = aie.tile(7, 3)
    %143 = aie.lock(%142, 2)
    %144 = aie.buffer(%142) {sym_name = "buf14"} : memref<64xi32, 2>
    %145 = aie.lock(%142, 1)
    %146 = aie.buffer(%142) {sym_name = "buf13"} : memref<64xi32, 2>
    %147 = aie.lock(%142, 0)
    %148 = aie.buffer(%142) {sym_name = "buf12"} : memref<64xi32, 2>
    %149 = aie.mem(%142)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%147, Acquire, 0)
      aie.dma_bd(%148 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%147, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%145, Acquire, 0)
      aie.dma_bd(%146 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%145, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%143, Acquire, 1)
      aie.dma_bd(%144 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%143, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %150 = aie.core(%142)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%147, Acquire, 1)
      aie.use_lock(%145, Acquire, 1)
      aie.use_lock(%143, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %148[%arg0] : memref<64xi32, 2>
        %201 = affine.load %146[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %144[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%143, Release, 1)
      aie.use_lock(%145, Release, 0)
      aie.use_lock(%147, Release, 0)
      cf.br ^bb1
    }
    %151 = aie.tile(7, 1)
    %152 = aie.tile(7, 0)
    %153 = aie.tile(10, 2)
    %154 = aie.lock(%153, 2)
    %155 = aie.buffer(%153) {sym_name = "buf11"} : memref<64xi32, 2>
    %156 = aie.lock(%153, 1)
    %157 = aie.buffer(%153) {sym_name = "buf10"} : memref<64xi32, 2>
    %158 = aie.lock(%153, 0)
    %159 = aie.buffer(%153) {sym_name = "buf9"} : memref<64xi32, 2>
    %160 = aie.mem(%153)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%158, Acquire, 0)
      aie.dma_bd(%159 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%158, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%156, Acquire, 0)
      aie.dma_bd(%157 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%156, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%154, Acquire, 1)
      aie.dma_bd(%155 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%154, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %161 = aie.core(%153)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%158, Acquire, 1)
      aie.use_lock(%156, Acquire, 1)
      aie.use_lock(%154, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %159[%arg0] : memref<64xi32, 2>
        %201 = affine.load %157[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %155[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%154, Release, 1)
      aie.use_lock(%156, Release, 0)
      aie.use_lock(%158, Release, 0)
      cf.br ^bb1
    }
    %162 = aie.tile(6, 2)
    %163 = aie.tile(6, 1)
    %164 = aie.tile(6, 0)
    %165 = aie.tile(9, 2)
    %166 = aie.lock(%165, 2)
    %167 = aie.buffer(%165) {sym_name = "buf8"} : memref<64xi32, 2>
    %168 = aie.lock(%165, 1)
    %169 = aie.buffer(%165) {sym_name = "buf7"} : memref<64xi32, 2>
    %170 = aie.lock(%165, 0)
    %171 = aie.buffer(%165) {sym_name = "buf6"} : memref<64xi32, 2>
    %172 = aie.mem(%165)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%170, Acquire, 0)
      aie.dma_bd(%171 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%170, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%168, Acquire, 0)
      aie.dma_bd(%169 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%168, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%166, Acquire, 1)
      aie.dma_bd(%167 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%166, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %173 = aie.core(%165)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%170, Acquire, 1)
      aie.use_lock(%168, Acquire, 1)
      aie.use_lock(%166, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %171[%arg0] : memref<64xi32, 2>
        %201 = affine.load %169[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %167[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%166, Release, 1)
      aie.use_lock(%168, Release, 0)
      aie.use_lock(%170, Release, 0)
      cf.br ^bb1
    }
    %174 = aie.tile(3, 2)
    %175 = aie.tile(3, 1)
    %176 = aie.tile(3, 0)
    %177 = aie.tile(1, 0)
    %178 = aie.tile(8, 2)
    %179 = aie.lock(%178, 2)
    %180 = aie.buffer(%178) {sym_name = "buf5"} : memref<64xi32, 2>
    %181 = aie.lock(%178, 1)
    %182 = aie.buffer(%178) {sym_name = "buf4"} : memref<64xi32, 2>
    %183 = aie.lock(%178, 0)
    %184 = aie.buffer(%178) {sym_name = "buf3"} : memref<64xi32, 2>
    %185 = aie.mem(%178)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%183, Acquire, 0)
      aie.dma_bd(%184 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%183, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%181, Acquire, 0)
      aie.dma_bd(%182 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%181, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%179, Acquire, 1)
      aie.dma_bd(%180 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%179, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %186 = aie.core(%178)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%183, Acquire, 1)
      aie.use_lock(%181, Acquire, 1)
      aie.use_lock(%179, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %184[%arg0] : memref<64xi32, 2>
        %201 = affine.load %182[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %180[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%179, Release, 1)
      aie.use_lock(%181, Release, 0)
      aie.use_lock(%183, Release, 0)
      cf.br ^bb1
    }
    %187 = aie.tile(2, 2)
    %188 = aie.tile(2, 1)
    %189 = aie.tile(2, 0)
    %190 = aie.tile(0, 0)
    %191 = aie.tile(7, 2)
    %192 = aie.lock(%191, 2)
    %193 = aie.buffer(%191) {sym_name = "buf2"} : memref<64xi32, 2>
    %194 = aie.lock(%191, 1)
    %195 = aie.buffer(%191) {sym_name = "buf1"} : memref<64xi32, 2>
    %196 = aie.lock(%191, 0)
    %197 = aie.buffer(%191) {sym_name = "buf0"} : memref<64xi32, 2>
    %198 = aie.mem(%191)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%196, Acquire, 0)
      aie.dma_bd(%197 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%196, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%194, Acquire, 0)
      aie.dma_bd(%195 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%194, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%192, Acquire, 1)
      aie.dma_bd(%193 : memref<64xi32, 2>) { len = 64 : i32 }
      aie.use_lock(%192, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %199 = aie.core(%191)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%196, Acquire, 1)
      aie.use_lock(%194, Acquire, 1)
      aie.use_lock(%192, Acquire, 0)
      affine.for %arg0 = 0 to 64 {
        %200 = affine.load %197[%arg0] : memref<64xi32, 2>
        %201 = affine.load %195[%arg0] : memref<64xi32, 2>
        %202 = arith.muli %200, %201 : i32
        affine.store %202, %193[%arg0] : memref<64xi32, 2>
      }
      aie.use_lock(%192, Release, 1)
      aie.use_lock(%194, Release, 0)
      aie.use_lock(%196, Release, 0)
      cf.br ^bb1
    }
    aie.flow(%189, DMA : 0, %191, DMA : 0)
    aie.flow(%189, DMA : 1, %191, DMA : 1)
    aie.flow(%191, DMA : 0, %189, DMA : 0)
    aie.flow(%176, DMA : 0, %178, DMA : 0)
    aie.flow(%176, DMA : 1, %178, DMA : 1)
    aie.flow(%178, DMA : 0, %189, DMA : 1)
    aie.flow(%164, DMA : 0, %165, DMA : 0)
    aie.flow(%164, DMA : 1, %165, DMA : 1)
    aie.flow(%165, DMA : 0, %176, DMA : 0)
    aie.flow(%152, DMA : 0, %153, DMA : 0)
    aie.flow(%152, DMA : 1, %153, DMA : 1)
    aie.flow(%153, DMA : 0, %176, DMA : 1)
    aie.flow(%140, DMA : 0, %142, DMA : 0)
    aie.flow(%140, DMA : 1, %142, DMA : 1)
    aie.flow(%142, DMA : 0, %164, DMA : 0)
    aie.flow(%128, DMA : 0, %130, DMA : 0)
    aie.flow(%128, DMA : 1, %130, DMA : 1)
    aie.flow(%130, DMA : 0, %164, DMA : 1)
    aie.flow(%116, DMA : 0, %117, DMA : 0)
    aie.flow(%116, DMA : 1, %117, DMA : 1)
    aie.flow(%117, DMA : 0, %152, DMA : 0)
    aie.flow(%104, DMA : 0, %105, DMA : 0)
    aie.flow(%104, DMA : 1, %105, DMA : 1)
    aie.flow(%105, DMA : 0, %152, DMA : 1)
    aie.flow(%91, DMA : 0, %93, DMA : 0)
    aie.flow(%91, DMA : 1, %93, DMA : 1)
    aie.flow(%93, DMA : 0, %140, DMA : 0)
    aie.flow(%78, DMA : 0, %80, DMA : 0)
    aie.flow(%78, DMA : 1, %80, DMA : 1)
    aie.flow(%80, DMA : 0, %140, DMA : 1)
    aie.flow(%66, DMA : 0, %67, DMA : 0)
    aie.flow(%66, DMA : 1, %67, DMA : 1)
    aie.flow(%67, DMA : 0, %128, DMA : 0)
    aie.flow(%54, DMA : 0, %55, DMA : 0)
    aie.flow(%54, DMA : 1, %55, DMA : 1)
    aie.flow(%55, DMA : 0, %128, DMA : 1)
    aie.flow(%41, DMA : 0, %43, DMA : 0)
    aie.flow(%41, DMA : 1, %43, DMA : 1)
    aie.flow(%43, DMA : 0, %116, DMA : 0)
    aie.flow(%28, DMA : 0, %30, DMA : 0)
    aie.flow(%28, DMA : 1, %30, DMA : 1)
    aie.flow(%30, DMA : 0, %116, DMA : 1)
    aie.flow(%15, DMA : 0, %17, DMA : 0)
    aie.flow(%15, DMA : 1, %17, DMA : 1)
    aie.flow(%17, DMA : 0, %104, DMA : 0)
    aie.flow(%2, DMA : 0, %4, DMA : 0)
    aie.flow(%2, DMA : 1, %4, DMA : 1)
    aie.flow(%4, DMA : 0, %104, DMA : 1)
  }
}
