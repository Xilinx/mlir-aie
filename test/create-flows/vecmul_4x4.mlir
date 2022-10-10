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
// CHECK: %[[T2:.*]] = AIE.tile(47, 0)
// CHECK: %[[T4:.*]] = AIE.tile(10, 5)
// CHECK: %[[T15:.*]] = AIE.tile(46, 0)
// CHECK: %[[T17:.*]] = AIE.tile(9, 5)
// CHECK: %[[T28:.*]] = AIE.tile(43, 0)
// CHECK: %[[T30:.*]] = AIE.tile(8, 5)
// CHECK: %[[T41:.*]] = AIE.tile(42, 0)
// CHECK: %[[T43:.*]] = AIE.tile(7, 5)
// CHECK: %[[T54:.*]] = AIE.tile(35, 0)
// CHECK: %[[T55:.*]] = AIE.tile(10, 4)
// CHECK: %[[T66:.*]] = AIE.tile(34, 0)
// CHECK: %[[T67:.*]] = AIE.tile(9, 4)
// CHECK: %[[T78:.*]] = AIE.tile(27, 0)
// CHECK: %[[T80:.*]] = AIE.tile(8, 4)
// CHECK: %[[T91:.*]] = AIE.tile(26, 0)
// CHECK: %[[T93:.*]] = AIE.tile(7, 4)
// CHECK: %[[T104:.*]] = AIE.tile(19, 0)
// CHECK: %[[T105:.*]] = AIE.tile(10, 3)
// CHECK: %[[T116:.*]] = AIE.tile(18, 0)
// CHECK: %[[T117:.*]] = AIE.tile(9, 3)
// CHECK: %[[T128:.*]] = AIE.tile(11, 0)
// CHECK: %[[T130:.*]] = AIE.tile(8, 3)
// CHECK: %[[T140:.*]] = AIE.tile(10, 0)
// CHECK: %[[T142:.*]] = AIE.tile(7, 3)
// CHECK: %[[T152:.*]] = AIE.tile(7, 0)
// CHECK: %[[T153:.*]] = AIE.tile(10, 2)
// CHECK: %[[T164:.*]] = AIE.tile(6, 0)
// CHECK: %[[T165:.*]] = AIE.tile(9, 2)
// CHECK: %[[T176:.*]] = AIE.tile(3, 0)
// CHECK: %[[T178:.*]] = AIE.tile(8, 2)
// CHECK: %[[T189:.*]] = AIE.tile(2, 0)
// CHECK: %[[T191:.*]] = AIE.tile(7, 2)

//
// CHECK: AIE.flow(%[[T2]], DMA : 0, %[[T4]], DMA : 0)
// CHECK: AIE.flow(%[[T2]], DMA : 1, %[[T4]], DMA : 1)
// CHECK: AIE.flow(%[[T4]], DMA : 0, %[[T104]], DMA : 1)
// CHECK: AIE.flow(%[[T15]], DMA : 0, %[[T17]], DMA : 0)
// CHECK: AIE.flow(%[[T15]], DMA : 1, %[[T17]], DMA : 1)
// CHECK: AIE.flow(%[[T17]], DMA : 0, %[[T104]], DMA : 0)
// CHECK: AIE.flow(%[[T28]], DMA : 0, %[[T30]], DMA : 0)
// CHECK: AIE.flow(%[[T28]], DMA : 1, %[[T30]], DMA : 1)
// CHECK: AIE.flow(%[[T30]], DMA : 0, %[[T116]], DMA : 1)
// CHECK: AIE.flow(%[[T41]], DMA : 0, %[[T43]], DMA : 0)
// CHECK: AIE.flow(%[[T41]], DMA : 1, %[[T43]], DMA : 1)
// CHECK: AIE.flow(%[[T43]], DMA : 0, %[[T116]], DMA : 0)
// CHECK: AIE.flow(%[[T54]], DMA : 0, %[[T55]], DMA : 0)
// CHECK: AIE.flow(%[[T54]], DMA : 1, %[[T55]], DMA : 1)
// CHECK: AIE.flow(%[[T55]], DMA : 0, %[[T128]], DMA : 1)
// CHECK: AIE.flow(%[[T66]], DMA : 0, %[[T67]], DMA : 0)
// CHECK: AIE.flow(%[[T66]], DMA : 1, %[[T67]], DMA : 1)
// CHECK: AIE.flow(%[[T67]], DMA : 0, %[[T128]], DMA : 0)
// CHECK: AIE.flow(%[[T78]], DMA : 0, %[[T80]], DMA : 0)
// CHECK: AIE.flow(%[[T78]], DMA : 1, %[[T80]], DMA : 1)
// CHECK: AIE.flow(%[[T80]], DMA : 0, %[[T140]], DMA : 1)
// CHECK: AIE.flow(%[[T91]], DMA : 0, %[[T93]], DMA : 0)
// CHECK: AIE.flow(%[[T91]], DMA : 1, %[[T93]], DMA : 1)
// CHECK: AIE.flow(%[[T93]], DMA : 0, %[[T140]], DMA : 0)
// CHECK: AIE.flow(%[[T104]], DMA : 0, %[[T105]], DMA : 0)
// CHECK: AIE.flow(%[[T104]], DMA : 1, %[[T105]], DMA : 1)
// CHECK: AIE.flow(%[[T105]], DMA : 0, %[[T152]], DMA : 1)
// CHECK: AIE.flow(%[[T116]], DMA : 0, %[[T117]], DMA : 0)
// CHECK: AIE.flow(%[[T116]], DMA : 1, %[[T117]], DMA : 1)
// CHECK: AIE.flow(%[[T117]], DMA : 0, %[[T152]], DMA : 0)
// CHECK: AIE.flow(%[[T128]], DMA : 0, %[[T130]], DMA : 0)
// CHECK: AIE.flow(%[[T128]], DMA : 1, %[[T130]], DMA : 1)
// CHECK: AIE.flow(%[[T130]], DMA : 0, %[[T164]], DMA : 1)
// CHECK: AIE.flow(%[[T140]], DMA : 0, %[[T142]], DMA : 0)
// CHECK: AIE.flow(%[[T140]], DMA : 1, %[[T142]], DMA : 1)
// CHECK: AIE.flow(%[[T142]], DMA : 0, %[[T164]], DMA : 0)
// CHECK: AIE.flow(%[[T152]], DMA : 0, %[[T153]], DMA : 0)
// CHECK: AIE.flow(%[[T152]], DMA : 1, %[[T153]], DMA : 1)
// CHECK: AIE.flow(%[[T153]], DMA : 0, %[[T176]], DMA : 1)
// CHECK: AIE.flow(%[[T164]], DMA : 0, %[[T165]], DMA : 0)
// CHECK: AIE.flow(%[[T164]], DMA : 1, %[[T165]], DMA : 1)
// CHECK: AIE.flow(%[[T165]], DMA : 0, %[[T176]], DMA : 0)
// CHECK: AIE.flow(%[[T176]], DMA : 0, %[[T178]], DMA : 0)
// CHECK: AIE.flow(%[[T176]], DMA : 1, %[[T178]], DMA : 1)
// CHECK: AIE.flow(%[[T178]], DMA : 0, %[[T189]], DMA : 1)
// CHECK: AIE.flow(%[[T189]], DMA : 0, %[[T191]], DMA : 0)
// CHECK: AIE.flow(%[[T189]], DMA : 1, %[[T191]], DMA : 1)
// CHECK: AIE.flow(%[[T191]], DMA : 0, %[[T189]], DMA : 0)

module @vecmul_4x4  {
  %0 = AIE.tile(47, 2)
  %1 = AIE.tile(47, 1)
  %2 = AIE.tile(47, 0)
  %3 = AIE.tile(3, 3)
  %4 = AIE.tile(10, 5)
  %5 = AIE.lock(%4, 2)
  %6 = AIE.buffer(%4) {sym_name = "buf47"} : memref<64xi32, 2>
  %7 = AIE.lock(%4, 1)
  %8 = AIE.buffer(%4) {sym_name = "buf46"} : memref<64xi32, 2>
  %9 = AIE.lock(%4, 0)
  %10 = AIE.buffer(%4) {sym_name = "buf45"} : memref<64xi32, 2>
  %11 = AIE.mem(%4)  {
    %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%9, Acquire, 0)
    AIE.dmaBd(<%10 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%9, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb4
    %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%7, Acquire, 0)
    AIE.dmaBd(<%8 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%7, Release, 1)
    cf.br ^bb3
  ^bb4:  // pred: ^bb0
    %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%5, Acquire, 1)
    AIE.dmaBd(<%6 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%5, Release, 0)
    cf.br ^bb5
  ^bb6:  // pred: ^bb2
    AIE.end
  }
  %12 = AIE.core(%4)  {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%9, Acquire, 1)
    AIE.useLock(%7, Acquire, 1)
    AIE.useLock(%5, Acquire, 0)
    affine.for %arg0 = 0 to 64 {
      %200 = affine.load %10[%arg0] : memref<64xi32, 2>
      %201 = affine.load %8[%arg0] : memref<64xi32, 2>
      %202 = arith.muli %200, %201 : i32
      affine.store %202, %6[%arg0] : memref<64xi32, 2>
    }
    AIE.useLock(%5, Release, 1)
    AIE.useLock(%7, Release, 0)
    AIE.useLock(%9, Release, 0)
    cf.br ^bb1
  }
  %13 = AIE.tile(46, 2)
  %14 = AIE.tile(46, 1)
  %15 = AIE.tile(46, 0)
  %16 = AIE.tile(2, 3)
  %17 = AIE.tile(9, 5)
  %18 = AIE.lock(%17, 2)
  %19 = AIE.buffer(%17) {sym_name = "buf44"} : memref<64xi32, 2>
  %20 = AIE.lock(%17, 1)
  %21 = AIE.buffer(%17) {sym_name = "buf43"} : memref<64xi32, 2>
  %22 = AIE.lock(%17, 0)
  %23 = AIE.buffer(%17) {sym_name = "buf42"} : memref<64xi32, 2>
  %24 = AIE.mem(%17)  {
    %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%22, Acquire, 0)
    AIE.dmaBd(<%23 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%22, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb4
    %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%20, Acquire, 0)
    AIE.dmaBd(<%21 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%20, Release, 1)
    cf.br ^bb3
  ^bb4:  // pred: ^bb0
    %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%18, Acquire, 1)
    AIE.dmaBd(<%19 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%18, Release, 0)
    cf.br ^bb5
  ^bb6:  // pred: ^bb2
    AIE.end
  }
  %25 = AIE.core(%17)  {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%22, Acquire, 1)
    AIE.useLock(%20, Acquire, 1)
    AIE.useLock(%18, Acquire, 0)
    affine.for %arg0 = 0 to 64 {
      %200 = affine.load %23[%arg0] : memref<64xi32, 2>
      %201 = affine.load %21[%arg0] : memref<64xi32, 2>
      %202 = arith.muli %200, %201 : i32
      affine.store %202, %19[%arg0] : memref<64xi32, 2>
    }
    AIE.useLock(%18, Release, 1)
    AIE.useLock(%20, Release, 0)
    AIE.useLock(%22, Release, 0)
    cf.br ^bb1
  }
  %26 = AIE.tile(43, 2)
  %27 = AIE.tile(43, 1)
  %28 = AIE.tile(43, 0)
  %29 = AIE.tile(1, 3)
  %30 = AIE.tile(8, 5)
  %31 = AIE.lock(%30, 2)
  %32 = AIE.buffer(%30) {sym_name = "buf41"} : memref<64xi32, 2>
  %33 = AIE.lock(%30, 1)
  %34 = AIE.buffer(%30) {sym_name = "buf40"} : memref<64xi32, 2>
  %35 = AIE.lock(%30, 0)
  %36 = AIE.buffer(%30) {sym_name = "buf39"} : memref<64xi32, 2>
  %37 = AIE.mem(%30)  {
    %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%35, Acquire, 0)
    AIE.dmaBd(<%36 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%35, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb4
    %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%33, Acquire, 0)
    AIE.dmaBd(<%34 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%33, Release, 1)
    cf.br ^bb3
  ^bb4:  // pred: ^bb0
    %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%31, Acquire, 1)
    AIE.dmaBd(<%32 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%31, Release, 0)
    cf.br ^bb5
  ^bb6:  // pred: ^bb2
    AIE.end
  }
  %38 = AIE.core(%30)  {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%35, Acquire, 1)
    AIE.useLock(%33, Acquire, 1)
    AIE.useLock(%31, Acquire, 0)
    affine.for %arg0 = 0 to 64 {
      %200 = affine.load %36[%arg0] : memref<64xi32, 2>
      %201 = affine.load %34[%arg0] : memref<64xi32, 2>
      %202 = arith.muli %200, %201 : i32
      affine.store %202, %32[%arg0] : memref<64xi32, 2>
    }
    AIE.useLock(%31, Release, 1)
    AIE.useLock(%33, Release, 0)
    AIE.useLock(%35, Release, 0)
    cf.br ^bb1
  }
  %39 = AIE.tile(42, 2)
  %40 = AIE.tile(42, 1)
  %41 = AIE.tile(42, 0)
  %42 = AIE.tile(0, 3)
  %43 = AIE.tile(7, 5)
  %44 = AIE.lock(%43, 2)
  %45 = AIE.buffer(%43) {sym_name = "buf38"} : memref<64xi32, 2>
  %46 = AIE.lock(%43, 1)
  %47 = AIE.buffer(%43) {sym_name = "buf37"} : memref<64xi32, 2>
  %48 = AIE.lock(%43, 0)
  %49 = AIE.buffer(%43) {sym_name = "buf36"} : memref<64xi32, 2>
  %50 = AIE.mem(%43)  {
    %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%48, Acquire, 0)
    AIE.dmaBd(<%49 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%48, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb4
    %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%46, Acquire, 0)
    AIE.dmaBd(<%47 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%46, Release, 1)
    cf.br ^bb3
  ^bb4:  // pred: ^bb0
    %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%44, Acquire, 1)
    AIE.dmaBd(<%45 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%44, Release, 0)
    cf.br ^bb5
  ^bb6:  // pred: ^bb2
    AIE.end
  }
  %51 = AIE.core(%43)  {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%48, Acquire, 1)
    AIE.useLock(%46, Acquire, 1)
    AIE.useLock(%44, Acquire, 0)
    affine.for %arg0 = 0 to 64 {
      %200 = affine.load %49[%arg0] : memref<64xi32, 2>
      %201 = affine.load %47[%arg0] : memref<64xi32, 2>
      %202 = arith.muli %200, %201 : i32
      affine.store %202, %45[%arg0] : memref<64xi32, 2>
    }
    AIE.useLock(%44, Release, 1)
    AIE.useLock(%46, Release, 0)
    AIE.useLock(%48, Release, 0)
    cf.br ^bb1
  }
  %52 = AIE.tile(35, 2)
  %53 = AIE.tile(35, 1)
  %54 = AIE.tile(35, 0)
  %55 = AIE.tile(10, 4)
  %56 = AIE.lock(%55, 2)
  %57 = AIE.buffer(%55) {sym_name = "buf35"} : memref<64xi32, 2>
  %58 = AIE.lock(%55, 1)
  %59 = AIE.buffer(%55) {sym_name = "buf34"} : memref<64xi32, 2>
  %60 = AIE.lock(%55, 0)
  %61 = AIE.buffer(%55) {sym_name = "buf33"} : memref<64xi32, 2>
  %62 = AIE.mem(%55)  {
    %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%60, Acquire, 0)
    AIE.dmaBd(<%61 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%60, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb4
    %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%58, Acquire, 0)
    AIE.dmaBd(<%59 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%58, Release, 1)
    cf.br ^bb3
  ^bb4:  // pred: ^bb0
    %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%56, Acquire, 1)
    AIE.dmaBd(<%57 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%56, Release, 0)
    cf.br ^bb5
  ^bb6:  // pred: ^bb2
    AIE.end
  }
  %63 = AIE.core(%55)  {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%60, Acquire, 1)
    AIE.useLock(%58, Acquire, 1)
    AIE.useLock(%56, Acquire, 0)
    affine.for %arg0 = 0 to 64 {
      %200 = affine.load %61[%arg0] : memref<64xi32, 2>
      %201 = affine.load %59[%arg0] : memref<64xi32, 2>
      %202 = arith.muli %200, %201 : i32
      affine.store %202, %57[%arg0] : memref<64xi32, 2>
    }
    AIE.useLock(%56, Release, 1)
    AIE.useLock(%58, Release, 0)
    AIE.useLock(%60, Release, 0)
    cf.br ^bb1
  }
  %64 = AIE.tile(34, 2)
  %65 = AIE.tile(34, 1)
  %66 = AIE.tile(34, 0)
  %67 = AIE.tile(9, 4)
  %68 = AIE.lock(%67, 2)
  %69 = AIE.buffer(%67) {sym_name = "buf32"} : memref<64xi32, 2>
  %70 = AIE.lock(%67, 1)
  %71 = AIE.buffer(%67) {sym_name = "buf31"} : memref<64xi32, 2>
  %72 = AIE.lock(%67, 0)
  %73 = AIE.buffer(%67) {sym_name = "buf30"} : memref<64xi32, 2>
  %74 = AIE.mem(%67)  {
    %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%72, Acquire, 0)
    AIE.dmaBd(<%73 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%72, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb4
    %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%70, Acquire, 0)
    AIE.dmaBd(<%71 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%70, Release, 1)
    cf.br ^bb3
  ^bb4:  // pred: ^bb0
    %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%68, Acquire, 1)
    AIE.dmaBd(<%69 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%68, Release, 0)
    cf.br ^bb5
  ^bb6:  // pred: ^bb2
    AIE.end
  }
  %75 = AIE.core(%67)  {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%72, Acquire, 1)
    AIE.useLock(%70, Acquire, 1)
    AIE.useLock(%68, Acquire, 0)
    affine.for %arg0 = 0 to 64 {
      %200 = affine.load %73[%arg0] : memref<64xi32, 2>
      %201 = affine.load %71[%arg0] : memref<64xi32, 2>
      %202 = arith.muli %200, %201 : i32
      affine.store %202, %69[%arg0] : memref<64xi32, 2>
    }
    AIE.useLock(%68, Release, 1)
    AIE.useLock(%70, Release, 0)
    AIE.useLock(%72, Release, 0)
    cf.br ^bb1
  }
  %76 = AIE.tile(27, 2)
  %77 = AIE.tile(27, 1)
  %78 = AIE.tile(27, 0)
  %79 = AIE.tile(1, 2)
  %80 = AIE.tile(8, 4)
  %81 = AIE.lock(%80, 2)
  %82 = AIE.buffer(%80) {sym_name = "buf29"} : memref<64xi32, 2>
  %83 = AIE.lock(%80, 1)
  %84 = AIE.buffer(%80) {sym_name = "buf28"} : memref<64xi32, 2>
  %85 = AIE.lock(%80, 0)
  %86 = AIE.buffer(%80) {sym_name = "buf27"} : memref<64xi32, 2>
  %87 = AIE.mem(%80)  {
    %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%85, Acquire, 0)
    AIE.dmaBd(<%86 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%85, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb4
    %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%83, Acquire, 0)
    AIE.dmaBd(<%84 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%83, Release, 1)
    cf.br ^bb3
  ^bb4:  // pred: ^bb0
    %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%81, Acquire, 1)
    AIE.dmaBd(<%82 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%81, Release, 0)
    cf.br ^bb5
  ^bb6:  // pred: ^bb2
    AIE.end
  }
  %88 = AIE.core(%80)  {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%85, Acquire, 1)
    AIE.useLock(%83, Acquire, 1)
    AIE.useLock(%81, Acquire, 0)
    affine.for %arg0 = 0 to 64 {
      %200 = affine.load %86[%arg0] : memref<64xi32, 2>
      %201 = affine.load %84[%arg0] : memref<64xi32, 2>
      %202 = arith.muli %200, %201 : i32
      affine.store %202, %82[%arg0] : memref<64xi32, 2>
    }
    AIE.useLock(%81, Release, 1)
    AIE.useLock(%83, Release, 0)
    AIE.useLock(%85, Release, 0)
    cf.br ^bb1
  }
  %89 = AIE.tile(26, 2)
  %90 = AIE.tile(26, 1)
  %91 = AIE.tile(26, 0)
  %92 = AIE.tile(0, 2)
  %93 = AIE.tile(7, 4)
  %94 = AIE.lock(%93, 2)
  %95 = AIE.buffer(%93) {sym_name = "buf26"} : memref<64xi32, 2>
  %96 = AIE.lock(%93, 1)
  %97 = AIE.buffer(%93) {sym_name = "buf25"} : memref<64xi32, 2>
  %98 = AIE.lock(%93, 0)
  %99 = AIE.buffer(%93) {sym_name = "buf24"} : memref<64xi32, 2>
  %100 = AIE.mem(%93)  {
    %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%98, Acquire, 0)
    AIE.dmaBd(<%99 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%98, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb4
    %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%96, Acquire, 0)
    AIE.dmaBd(<%97 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%96, Release, 1)
    cf.br ^bb3
  ^bb4:  // pred: ^bb0
    %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%94, Acquire, 1)
    AIE.dmaBd(<%95 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%94, Release, 0)
    cf.br ^bb5
  ^bb6:  // pred: ^bb2
    AIE.end
  }
  %101 = AIE.core(%93)  {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%98, Acquire, 1)
    AIE.useLock(%96, Acquire, 1)
    AIE.useLock(%94, Acquire, 0)
    affine.for %arg0 = 0 to 64 {
      %200 = affine.load %99[%arg0] : memref<64xi32, 2>
      %201 = affine.load %97[%arg0] : memref<64xi32, 2>
      %202 = arith.muli %200, %201 : i32
      affine.store %202, %95[%arg0] : memref<64xi32, 2>
    }
    AIE.useLock(%94, Release, 1)
    AIE.useLock(%96, Release, 0)
    AIE.useLock(%98, Release, 0)
    cf.br ^bb1
  }
  %102 = AIE.tile(19, 2)
  %103 = AIE.tile(19, 1)
  %104 = AIE.tile(19, 0)
  %105 = AIE.tile(10, 3)
  %106 = AIE.lock(%105, 2)
  %107 = AIE.buffer(%105) {sym_name = "buf23"} : memref<64xi32, 2>
  %108 = AIE.lock(%105, 1)
  %109 = AIE.buffer(%105) {sym_name = "buf22"} : memref<64xi32, 2>
  %110 = AIE.lock(%105, 0)
  %111 = AIE.buffer(%105) {sym_name = "buf21"} : memref<64xi32, 2>
  %112 = AIE.mem(%105)  {
    %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%110, Acquire, 0)
    AIE.dmaBd(<%111 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%110, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb4
    %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%108, Acquire, 0)
    AIE.dmaBd(<%109 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%108, Release, 1)
    cf.br ^bb3
  ^bb4:  // pred: ^bb0
    %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%106, Acquire, 1)
    AIE.dmaBd(<%107 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%106, Release, 0)
    cf.br ^bb5
  ^bb6:  // pred: ^bb2
    AIE.end
  }
  %113 = AIE.core(%105)  {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%110, Acquire, 1)
    AIE.useLock(%108, Acquire, 1)
    AIE.useLock(%106, Acquire, 0)
    affine.for %arg0 = 0 to 64 {
      %200 = affine.load %111[%arg0] : memref<64xi32, 2>
      %201 = affine.load %109[%arg0] : memref<64xi32, 2>
      %202 = arith.muli %200, %201 : i32
      affine.store %202, %107[%arg0] : memref<64xi32, 2>
    }
    AIE.useLock(%106, Release, 1)
    AIE.useLock(%108, Release, 0)
    AIE.useLock(%110, Release, 0)
    cf.br ^bb1
  }
  %114 = AIE.tile(18, 2)
  %115 = AIE.tile(18, 1)
  %116 = AIE.tile(18, 0)
  %117 = AIE.tile(9, 3)
  %118 = AIE.lock(%117, 2)
  %119 = AIE.buffer(%117) {sym_name = "buf20"} : memref<64xi32, 2>
  %120 = AIE.lock(%117, 1)
  %121 = AIE.buffer(%117) {sym_name = "buf19"} : memref<64xi32, 2>
  %122 = AIE.lock(%117, 0)
  %123 = AIE.buffer(%117) {sym_name = "buf18"} : memref<64xi32, 2>
  %124 = AIE.mem(%117)  {
    %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%122, Acquire, 0)
    AIE.dmaBd(<%123 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%122, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb4
    %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%120, Acquire, 0)
    AIE.dmaBd(<%121 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%120, Release, 1)
    cf.br ^bb3
  ^bb4:  // pred: ^bb0
    %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%118, Acquire, 1)
    AIE.dmaBd(<%119 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%118, Release, 0)
    cf.br ^bb5
  ^bb6:  // pred: ^bb2
    AIE.end
  }
  %125 = AIE.core(%117)  {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%122, Acquire, 1)
    AIE.useLock(%120, Acquire, 1)
    AIE.useLock(%118, Acquire, 0)
    affine.for %arg0 = 0 to 64 {
      %200 = affine.load %123[%arg0] : memref<64xi32, 2>
      %201 = affine.load %121[%arg0] : memref<64xi32, 2>
      %202 = arith.muli %200, %201 : i32
      affine.store %202, %119[%arg0] : memref<64xi32, 2>
    }
    AIE.useLock(%118, Release, 1)
    AIE.useLock(%120, Release, 0)
    AIE.useLock(%122, Release, 0)
    cf.br ^bb1
  }
  %126 = AIE.tile(11, 2)
  %127 = AIE.tile(11, 1)
  %128 = AIE.tile(11, 0)
  %129 = AIE.tile(1, 1)
  %130 = AIE.tile(8, 3)
  %131 = AIE.lock(%130, 2)
  %132 = AIE.buffer(%130) {sym_name = "buf17"} : memref<64xi32, 2>
  %133 = AIE.lock(%130, 1)
  %134 = AIE.buffer(%130) {sym_name = "buf16"} : memref<64xi32, 2>
  %135 = AIE.lock(%130, 0)
  %136 = AIE.buffer(%130) {sym_name = "buf15"} : memref<64xi32, 2>
  %137 = AIE.mem(%130)  {
    %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%135, Acquire, 0)
    AIE.dmaBd(<%136 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%135, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb4
    %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%133, Acquire, 0)
    AIE.dmaBd(<%134 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%133, Release, 1)
    cf.br ^bb3
  ^bb4:  // pred: ^bb0
    %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%131, Acquire, 1)
    AIE.dmaBd(<%132 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%131, Release, 0)
    cf.br ^bb5
  ^bb6:  // pred: ^bb2
    AIE.end
  }
  %138 = AIE.core(%130)  {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%135, Acquire, 1)
    AIE.useLock(%133, Acquire, 1)
    AIE.useLock(%131, Acquire, 0)
    affine.for %arg0 = 0 to 64 {
      %200 = affine.load %136[%arg0] : memref<64xi32, 2>
      %201 = affine.load %134[%arg0] : memref<64xi32, 2>
      %202 = arith.muli %200, %201 : i32
      affine.store %202, %132[%arg0] : memref<64xi32, 2>
    }
    AIE.useLock(%131, Release, 1)
    AIE.useLock(%133, Release, 0)
    AIE.useLock(%135, Release, 0)
    cf.br ^bb1
  }
  %139 = AIE.tile(10, 1)
  %140 = AIE.tile(10, 0)
  %141 = AIE.tile(0, 1)
  %142 = AIE.tile(7, 3)
  %143 = AIE.lock(%142, 2)
  %144 = AIE.buffer(%142) {sym_name = "buf14"} : memref<64xi32, 2>
  %145 = AIE.lock(%142, 1)
  %146 = AIE.buffer(%142) {sym_name = "buf13"} : memref<64xi32, 2>
  %147 = AIE.lock(%142, 0)
  %148 = AIE.buffer(%142) {sym_name = "buf12"} : memref<64xi32, 2>
  %149 = AIE.mem(%142)  {
    %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%147, Acquire, 0)
    AIE.dmaBd(<%148 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%147, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb4
    %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%145, Acquire, 0)
    AIE.dmaBd(<%146 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%145, Release, 1)
    cf.br ^bb3
  ^bb4:  // pred: ^bb0
    %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%143, Acquire, 1)
    AIE.dmaBd(<%144 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%143, Release, 0)
    cf.br ^bb5
  ^bb6:  // pred: ^bb2
    AIE.end
  }
  %150 = AIE.core(%142)  {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%147, Acquire, 1)
    AIE.useLock(%145, Acquire, 1)
    AIE.useLock(%143, Acquire, 0)
    affine.for %arg0 = 0 to 64 {
      %200 = affine.load %148[%arg0] : memref<64xi32, 2>
      %201 = affine.load %146[%arg0] : memref<64xi32, 2>
      %202 = arith.muli %200, %201 : i32
      affine.store %202, %144[%arg0] : memref<64xi32, 2>
    }
    AIE.useLock(%143, Release, 1)
    AIE.useLock(%145, Release, 0)
    AIE.useLock(%147, Release, 0)
    cf.br ^bb1
  }
  %151 = AIE.tile(7, 1)
  %152 = AIE.tile(7, 0)
  %153 = AIE.tile(10, 2)
  %154 = AIE.lock(%153, 2)
  %155 = AIE.buffer(%153) {sym_name = "buf11"} : memref<64xi32, 2>
  %156 = AIE.lock(%153, 1)
  %157 = AIE.buffer(%153) {sym_name = "buf10"} : memref<64xi32, 2>
  %158 = AIE.lock(%153, 0)
  %159 = AIE.buffer(%153) {sym_name = "buf9"} : memref<64xi32, 2>
  %160 = AIE.mem(%153)  {
    %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%158, Acquire, 0)
    AIE.dmaBd(<%159 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%158, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb4
    %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%156, Acquire, 0)
    AIE.dmaBd(<%157 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%156, Release, 1)
    cf.br ^bb3
  ^bb4:  // pred: ^bb0
    %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%154, Acquire, 1)
    AIE.dmaBd(<%155 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%154, Release, 0)
    cf.br ^bb5
  ^bb6:  // pred: ^bb2
    AIE.end
  }
  %161 = AIE.core(%153)  {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%158, Acquire, 1)
    AIE.useLock(%156, Acquire, 1)
    AIE.useLock(%154, Acquire, 0)
    affine.for %arg0 = 0 to 64 {
      %200 = affine.load %159[%arg0] : memref<64xi32, 2>
      %201 = affine.load %157[%arg0] : memref<64xi32, 2>
      %202 = arith.muli %200, %201 : i32
      affine.store %202, %155[%arg0] : memref<64xi32, 2>
    }
    AIE.useLock(%154, Release, 1)
    AIE.useLock(%156, Release, 0)
    AIE.useLock(%158, Release, 0)
    cf.br ^bb1
  }
  %162 = AIE.tile(6, 2)
  %163 = AIE.tile(6, 1)
  %164 = AIE.tile(6, 0)
  %165 = AIE.tile(9, 2)
  %166 = AIE.lock(%165, 2)
  %167 = AIE.buffer(%165) {sym_name = "buf8"} : memref<64xi32, 2>
  %168 = AIE.lock(%165, 1)
  %169 = AIE.buffer(%165) {sym_name = "buf7"} : memref<64xi32, 2>
  %170 = AIE.lock(%165, 0)
  %171 = AIE.buffer(%165) {sym_name = "buf6"} : memref<64xi32, 2>
  %172 = AIE.mem(%165)  {
    %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%170, Acquire, 0)
    AIE.dmaBd(<%171 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%170, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb4
    %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%168, Acquire, 0)
    AIE.dmaBd(<%169 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%168, Release, 1)
    cf.br ^bb3
  ^bb4:  // pred: ^bb0
    %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%166, Acquire, 1)
    AIE.dmaBd(<%167 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%166, Release, 0)
    cf.br ^bb5
  ^bb6:  // pred: ^bb2
    AIE.end
  }
  %173 = AIE.core(%165)  {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%170, Acquire, 1)
    AIE.useLock(%168, Acquire, 1)
    AIE.useLock(%166, Acquire, 0)
    affine.for %arg0 = 0 to 64 {
      %200 = affine.load %171[%arg0] : memref<64xi32, 2>
      %201 = affine.load %169[%arg0] : memref<64xi32, 2>
      %202 = arith.muli %200, %201 : i32
      affine.store %202, %167[%arg0] : memref<64xi32, 2>
    }
    AIE.useLock(%166, Release, 1)
    AIE.useLock(%168, Release, 0)
    AIE.useLock(%170, Release, 0)
    cf.br ^bb1
  }
  %174 = AIE.tile(3, 2)
  %175 = AIE.tile(3, 1)
  %176 = AIE.tile(3, 0)
  %177 = AIE.tile(1, 0)
  %178 = AIE.tile(8, 2)
  %179 = AIE.lock(%178, 2)
  %180 = AIE.buffer(%178) {sym_name = "buf5"} : memref<64xi32, 2>
  %181 = AIE.lock(%178, 1)
  %182 = AIE.buffer(%178) {sym_name = "buf4"} : memref<64xi32, 2>
  %183 = AIE.lock(%178, 0)
  %184 = AIE.buffer(%178) {sym_name = "buf3"} : memref<64xi32, 2>
  %185 = AIE.mem(%178)  {
    %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%183, Acquire, 0)
    AIE.dmaBd(<%184 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%183, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb4
    %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%181, Acquire, 0)
    AIE.dmaBd(<%182 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%181, Release, 1)
    cf.br ^bb3
  ^bb4:  // pred: ^bb0
    %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%179, Acquire, 1)
    AIE.dmaBd(<%180 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%179, Release, 0)
    cf.br ^bb5
  ^bb6:  // pred: ^bb2
    AIE.end
  }
  %186 = AIE.core(%178)  {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%183, Acquire, 1)
    AIE.useLock(%181, Acquire, 1)
    AIE.useLock(%179, Acquire, 0)
    affine.for %arg0 = 0 to 64 {
      %200 = affine.load %184[%arg0] : memref<64xi32, 2>
      %201 = affine.load %182[%arg0] : memref<64xi32, 2>
      %202 = arith.muli %200, %201 : i32
      affine.store %202, %180[%arg0] : memref<64xi32, 2>
    }
    AIE.useLock(%179, Release, 1)
    AIE.useLock(%181, Release, 0)
    AIE.useLock(%183, Release, 0)
    cf.br ^bb1
  }
  %187 = AIE.tile(2, 2)
  %188 = AIE.tile(2, 1)
  %189 = AIE.tile(2, 0)
  %190 = AIE.tile(0, 0)
  %191 = AIE.tile(7, 2)
  %192 = AIE.lock(%191, 2)
  %193 = AIE.buffer(%191) {sym_name = "buf2"} : memref<64xi32, 2>
  %194 = AIE.lock(%191, 1)
  %195 = AIE.buffer(%191) {sym_name = "buf1"} : memref<64xi32, 2>
  %196 = AIE.lock(%191, 0)
  %197 = AIE.buffer(%191) {sym_name = "buf0"} : memref<64xi32, 2>
  %198 = AIE.mem(%191)  {
    %200 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%196, Acquire, 0)
    AIE.dmaBd(<%197 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%196, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb4
    %201 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb6)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%194, Acquire, 0)
    AIE.dmaBd(<%195 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%194, Release, 1)
    cf.br ^bb3
  ^bb4:  // pred: ^bb0
    %202 = AIE.dmaStart(MM2S, 0, ^bb5, ^bb2)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%192, Acquire, 1)
    AIE.dmaBd(<%193 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%192, Release, 0)
    cf.br ^bb5
  ^bb6:  // pred: ^bb2
    AIE.end
  }
  %199 = AIE.core(%191)  {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%196, Acquire, 1)
    AIE.useLock(%194, Acquire, 1)
    AIE.useLock(%192, Acquire, 0)
    affine.for %arg0 = 0 to 64 {
      %200 = affine.load %197[%arg0] : memref<64xi32, 2>
      %201 = affine.load %195[%arg0] : memref<64xi32, 2>
      %202 = arith.muli %200, %201 : i32
      affine.store %202, %193[%arg0] : memref<64xi32, 2>
    }
    AIE.useLock(%192, Release, 1)
    AIE.useLock(%194, Release, 0)
    AIE.useLock(%196, Release, 0)
    cf.br ^bb1
  }
  AIE.flow(%189, DMA : 0, %191, DMA : 0)
  AIE.flow(%189, DMA : 1, %191, DMA : 1)
  AIE.flow(%191, DMA : 0, %189, DMA : 0)
  AIE.flow(%176, DMA : 0, %178, DMA : 0)
  AIE.flow(%176, DMA : 1, %178, DMA : 1)
  AIE.flow(%178, DMA : 0, %189, DMA : 1)
  AIE.flow(%164, DMA : 0, %165, DMA : 0)
  AIE.flow(%164, DMA : 1, %165, DMA : 1)
  AIE.flow(%165, DMA : 0, %176, DMA : 0)
  AIE.flow(%152, DMA : 0, %153, DMA : 0)
  AIE.flow(%152, DMA : 1, %153, DMA : 1)
  AIE.flow(%153, DMA : 0, %176, DMA : 1)
  AIE.flow(%140, DMA : 0, %142, DMA : 0)
  AIE.flow(%140, DMA : 1, %142, DMA : 1)
  AIE.flow(%142, DMA : 0, %164, DMA : 0)
  AIE.flow(%128, DMA : 0, %130, DMA : 0)
  AIE.flow(%128, DMA : 1, %130, DMA : 1)
  AIE.flow(%130, DMA : 0, %164, DMA : 1)
  AIE.flow(%116, DMA : 0, %117, DMA : 0)
  AIE.flow(%116, DMA : 1, %117, DMA : 1)
  AIE.flow(%117, DMA : 0, %152, DMA : 0)
  AIE.flow(%104, DMA : 0, %105, DMA : 0)
  AIE.flow(%104, DMA : 1, %105, DMA : 1)
  AIE.flow(%105, DMA : 0, %152, DMA : 1)
  AIE.flow(%91, DMA : 0, %93, DMA : 0)
  AIE.flow(%91, DMA : 1, %93, DMA : 1)
  AIE.flow(%93, DMA : 0, %140, DMA : 0)
  AIE.flow(%78, DMA : 0, %80, DMA : 0)
  AIE.flow(%78, DMA : 1, %80, DMA : 1)
  AIE.flow(%80, DMA : 0, %140, DMA : 1)
  AIE.flow(%66, DMA : 0, %67, DMA : 0)
  AIE.flow(%66, DMA : 1, %67, DMA : 1)
  AIE.flow(%67, DMA : 0, %128, DMA : 0)
  AIE.flow(%54, DMA : 0, %55, DMA : 0)
  AIE.flow(%54, DMA : 1, %55, DMA : 1)
  AIE.flow(%55, DMA : 0, %128, DMA : 1)
  AIE.flow(%41, DMA : 0, %43, DMA : 0)
  AIE.flow(%41, DMA : 1, %43, DMA : 1)
  AIE.flow(%43, DMA : 0, %116, DMA : 0)
  AIE.flow(%28, DMA : 0, %30, DMA : 0)
  AIE.flow(%28, DMA : 1, %30, DMA : 1)
  AIE.flow(%30, DMA : 0, %116, DMA : 1)
  AIE.flow(%15, DMA : 0, %17, DMA : 0)
  AIE.flow(%15, DMA : 1, %17, DMA : 1)
  AIE.flow(%17, DMA : 0, %104, DMA : 0)
  AIE.flow(%2, DMA : 0, %4, DMA : 0)
  AIE.flow(%2, DMA : 1, %4, DMA : 1)
  AIE.flow(%4, DMA : 0, %104, DMA : 1)
}
