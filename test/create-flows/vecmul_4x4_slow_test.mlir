//===- vecmul_4x4.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows=remove-lifted=false %s -o %t.opt
// RUN: FileCheck %s --check-prefix=CHECK1 < %t.opt
// RUN: aie-translate --aie-flows-to-json %t.opt | FileCheck %s --check-prefix=CHECK2

// CHECK1: %[[T2:.*]] = aie.tile(47, 0)
// CHECK1: %[[T4:.*]] = aie.tile(10, 5)
// CHECK1: %[[T15:.*]] = aie.tile(46, 0)
// CHECK1: %[[T17:.*]] = aie.tile(9, 5)
// CHECK1: %[[T28:.*]] = aie.tile(43, 0)
// CHECK1: %[[T30:.*]] = aie.tile(8, 5)
// CHECK1: %[[T41:.*]] = aie.tile(42, 0)
// CHECK1: %[[T43:.*]] = aie.tile(7, 5)
// CHECK1: %[[T54:.*]] = aie.tile(35, 0)
// CHECK1: %[[T55:.*]] = aie.tile(10, 4)
// CHECK1: %[[T66:.*]] = aie.tile(34, 0)
// CHECK1: %[[T67:.*]] = aie.tile(9, 4)
// CHECK1: %[[T78:.*]] = aie.tile(27, 0)
// CHECK1: %[[T80:.*]] = aie.tile(8, 4)
// CHECK1: %[[T91:.*]] = aie.tile(26, 0)
// CHECK1: %[[T93:.*]] = aie.tile(7, 4)
// CHECK1: %[[T104:.*]] = aie.tile(19, 0)
// CHECK1: %[[T105:.*]] = aie.tile(10, 3)
// CHECK1: %[[T116:.*]] = aie.tile(18, 0)
// CHECK1: %[[T117:.*]] = aie.tile(9, 3)
// CHECK1: %[[T128:.*]] = aie.tile(11, 0)
// CHECK1: %[[T130:.*]] = aie.tile(8, 3)
// CHECK1: %[[T140:.*]] = aie.tile(10, 0)
// CHECK1: %[[T142:.*]] = aie.tile(7, 3)
// CHECK1: %[[T152:.*]] = aie.tile(7, 0)
// CHECK1: %[[T153:.*]] = aie.tile(10, 2)
// CHECK1: %[[T164:.*]] = aie.tile(6, 0)
// CHECK1: %[[T165:.*]] = aie.tile(9, 2)
// CHECK1: %[[T176:.*]] = aie.tile(3, 0)
// CHECK1: %[[T178:.*]] = aie.tile(8, 2)
// CHECK1: %[[T189:.*]] = aie.tile(2, 0)
// CHECK1: %[[T191:.*]] = aie.tile(7, 2)

//
// CHECK1: aie.flow(%[[T2]], DMA : 0, %[[T4]], DMA : 0)
// CHECK1: aie.flow(%[[T2]], DMA : 1, %[[T4]], DMA : 1)
// CHECK1: aie.flow(%[[T4]], DMA : 0, %[[T104]], DMA : 1)
// CHECK1: aie.flow(%[[T15]], DMA : 0, %[[T17]], DMA : 0)
// CHECK1: aie.flow(%[[T15]], DMA : 1, %[[T17]], DMA : 1)
// CHECK1: aie.flow(%[[T17]], DMA : 0, %[[T104]], DMA : 0)
// CHECK1: aie.flow(%[[T28]], DMA : 0, %[[T30]], DMA : 0)
// CHECK1: aie.flow(%[[T28]], DMA : 1, %[[T30]], DMA : 1)
// CHECK1: aie.flow(%[[T30]], DMA : 0, %[[T116]], DMA : 1)
// CHECK1: aie.flow(%[[T41]], DMA : 0, %[[T43]], DMA : 0)
// CHECK1: aie.flow(%[[T41]], DMA : 1, %[[T43]], DMA : 1)
// CHECK1: aie.flow(%[[T43]], DMA : 0, %[[T116]], DMA : 0)
// CHECK1: aie.flow(%[[T54]], DMA : 0, %[[T55]], DMA : 0)
// CHECK1: aie.flow(%[[T54]], DMA : 1, %[[T55]], DMA : 1)
// CHECK1: aie.flow(%[[T55]], DMA : 0, %[[T128]], DMA : 1)
// CHECK1: aie.flow(%[[T66]], DMA : 0, %[[T67]], DMA : 0)
// CHECK1: aie.flow(%[[T66]], DMA : 1, %[[T67]], DMA : 1)
// CHECK1: aie.flow(%[[T67]], DMA : 0, %[[T128]], DMA : 0)
// CHECK1: aie.flow(%[[T78]], DMA : 0, %[[T80]], DMA : 0)
// CHECK1: aie.flow(%[[T78]], DMA : 1, %[[T80]], DMA : 1)
// CHECK1: aie.flow(%[[T80]], DMA : 0, %[[T140]], DMA : 1)
// CHECK1: aie.flow(%[[T91]], DMA : 0, %[[T93]], DMA : 0)
// CHECK1: aie.flow(%[[T91]], DMA : 1, %[[T93]], DMA : 1)
// CHECK1: aie.flow(%[[T93]], DMA : 0, %[[T140]], DMA : 0)
// CHECK1: aie.flow(%[[T104]], DMA : 0, %[[T105]], DMA : 0)
// CHECK1: aie.flow(%[[T104]], DMA : 1, %[[T105]], DMA : 1)
// CHECK1: aie.flow(%[[T105]], DMA : 0, %[[T152]], DMA : 1)
// CHECK1: aie.flow(%[[T116]], DMA : 0, %[[T117]], DMA : 0)
// CHECK1: aie.flow(%[[T116]], DMA : 1, %[[T117]], DMA : 1)
// CHECK1: aie.flow(%[[T117]], DMA : 0, %[[T152]], DMA : 0)
// CHECK1: aie.flow(%[[T128]], DMA : 0, %[[T130]], DMA : 0)
// CHECK1: aie.flow(%[[T128]], DMA : 1, %[[T130]], DMA : 1)
// CHECK1: aie.flow(%[[T130]], DMA : 0, %[[T164]], DMA : 1)
// CHECK1: aie.flow(%[[T140]], DMA : 0, %[[T142]], DMA : 0)
// CHECK1: aie.flow(%[[T140]], DMA : 1, %[[T142]], DMA : 1)
// CHECK1: aie.flow(%[[T142]], DMA : 0, %[[T164]], DMA : 0)
// CHECK1: aie.flow(%[[T152]], DMA : 0, %[[T153]], DMA : 0)
// CHECK1: aie.flow(%[[T152]], DMA : 1, %[[T153]], DMA : 1)
// CHECK1: aie.flow(%[[T153]], DMA : 0, %[[T176]], DMA : 1)
// CHECK1: aie.flow(%[[T164]], DMA : 0, %[[T165]], DMA : 0)
// CHECK1: aie.flow(%[[T164]], DMA : 1, %[[T165]], DMA : 1)
// CHECK1: aie.flow(%[[T165]], DMA : 0, %[[T176]], DMA : 0)
// CHECK1: aie.flow(%[[T176]], DMA : 0, %[[T178]], DMA : 0)
// CHECK1: aie.flow(%[[T176]], DMA : 1, %[[T178]], DMA : 1)
// CHECK1: aie.flow(%[[T178]], DMA : 0, %[[T189]], DMA : 1)
// CHECK1: aie.flow(%[[T189]], DMA : 0, %[[T191]], DMA : 0)
// CHECK1: aie.flow(%[[T189]], DMA : 1, %[[T191]], DMA : 1)
// CHECK1: aie.flow(%[[T191]], DMA : 0, %[[T189]], DMA : 0)

// CHECK2: "total_path_length": 792

module @vecmul_4x4  {
  aie.device(xcvc1902) {
    %0 = aie.tile(47, 2)
    %1 = aie.tile(47, 1)
    %2 = aie.tile(47, 0)
    %3 = aie.tile(3, 3)
    %4 = aie.tile(10, 5)
    aie.lock @lock_5(%4, 2)
    aie.buffer @buf47(%4) : memref<64xi32, 2>
    aie.lock @lock_7(%4, 1)
    aie.buffer @buf46(%4) : memref<64xi32, 2>
    aie.lock @lock_9(%4, 0)
    aie.buffer @buf45(%4) : memref<64xi32, 2>
    %11 = aie.mem(%4)  {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c0_ul1 = arith.constant 0 : i32
      aie.use_lock(@lock_9, Acquire, %c0_ul1)
      %buffer_10_0 = aie.get_buffer @buf45 : memref<64xi32, 2>
      aie.dma_bd(%buffer_10_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(@lock_9, Release, %c1_ul2)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      %c0_ul3 = arith.constant 0 : i32
      aie.use_lock(@lock_7, Acquire, %c0_ul3)
      %buffer_8_0 = aie.get_buffer @buf46 : memref<64xi32, 2>
      aie.dma_bd(%buffer_8_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(@lock_7, Release, %c1_ul4)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      %c1_ul5 = arith.constant 1 : i32
      aie.use_lock(@lock_5, Acquire, %c1_ul5)
      %buffer_6_0 = aie.get_buffer @buf47 : memref<64xi32, 2>
      aie.dma_bd(%buffer_6_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c0_ul6 = arith.constant 0 : i32
      aie.use_lock(@lock_5, Release, %c0_ul6)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %12 = aie.core(%4)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul7 = arith.constant 1 : i32
      aie.use_lock(@lock_9, Acquire, %c1_ul7)
      %c1_ul8 = arith.constant 1 : i32
      aie.use_lock(@lock_7, Acquire, %c1_ul8)
      %c0_ul9 = arith.constant 0 : i32
      aie.use_lock(@lock_5, Acquire, %c0_ul9)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %10[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %8[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %6[%arg0] : memref<64xi32, 2>
      // }
      %c1_ul10 = arith.constant 1 : i32
      aie.use_lock(@lock_5, Release, %c1_ul10)
      %c0_ul11 = arith.constant 0 : i32
      aie.use_lock(@lock_7, Release, %c0_ul11)
      %c0_ul12 = arith.constant 0 : i32
      aie.use_lock(@lock_9, Release, %c0_ul12)
      cf.br ^bb1
    }
    %13 = aie.tile(46, 2)
    %14 = aie.tile(46, 1)
    %15 = aie.tile(46, 0)
    %16 = aie.tile(2, 3)
    %17 = aie.tile(9, 5)
    aie.lock @lock_18(%17, 2)
    aie.buffer @buf44(%17) : memref<64xi32, 2>
    aie.lock @lock_20(%17, 1)
    aie.buffer @buf43(%17) : memref<64xi32, 2>
    aie.lock @lock_22(%17, 0)
    aie.buffer @buf42(%17) : memref<64xi32, 2>
    %24 = aie.mem(%17)  {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c0_ul13 = arith.constant 0 : i32
      aie.use_lock(@lock_22, Acquire, %c0_ul13)
      %buffer_23_0 = aie.get_buffer @buf42 : memref<64xi32, 2>
      aie.dma_bd(%buffer_23_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul14 = arith.constant 1 : i32
      aie.use_lock(@lock_22, Release, %c1_ul14)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      %c0_ul15 = arith.constant 0 : i32
      aie.use_lock(@lock_20, Acquire, %c0_ul15)
      %buffer_21_0 = aie.get_buffer @buf43 : memref<64xi32, 2>
      aie.dma_bd(%buffer_21_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul16 = arith.constant 1 : i32
      aie.use_lock(@lock_20, Release, %c1_ul16)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      %c1_ul17 = arith.constant 1 : i32
      aie.use_lock(@lock_18, Acquire, %c1_ul17)
      %buffer_19_0 = aie.get_buffer @buf44 : memref<64xi32, 2>
      aie.dma_bd(%buffer_19_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c0_ul18 = arith.constant 0 : i32
      aie.use_lock(@lock_18, Release, %c0_ul18)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %25 = aie.core(%17)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul19 = arith.constant 1 : i32
      aie.use_lock(@lock_22, Acquire, %c1_ul19)
      %c1_ul20 = arith.constant 1 : i32
      aie.use_lock(@lock_20, Acquire, %c1_ul20)
      %c0_ul21 = arith.constant 0 : i32
      aie.use_lock(@lock_18, Acquire, %c0_ul21)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %23[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %21[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %19[%arg0] : memref<64xi32, 2>
      // }
      %c1_ul22 = arith.constant 1 : i32
      aie.use_lock(@lock_18, Release, %c1_ul22)
      %c0_ul23 = arith.constant 0 : i32
      aie.use_lock(@lock_20, Release, %c0_ul23)
      %c0_ul24 = arith.constant 0 : i32
      aie.use_lock(@lock_22, Release, %c0_ul24)
      cf.br ^bb1
    }
    %26 = aie.tile(43, 2)
    %27 = aie.tile(43, 1)
    %28 = aie.tile(43, 0)
    %29 = aie.tile(1, 3)
    %30 = aie.tile(8, 5)
    aie.lock @lock_31(%30, 2)
    aie.buffer @buf41(%30) : memref<64xi32, 2>
    aie.lock @lock_33(%30, 1)
    aie.buffer @buf40(%30) : memref<64xi32, 2>
    aie.lock @lock_35(%30, 0)
    aie.buffer @buf39(%30) : memref<64xi32, 2>
    %37 = aie.mem(%30)  {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c0_ul25 = arith.constant 0 : i32
      aie.use_lock(@lock_35, Acquire, %c0_ul25)
      %buffer_36_0 = aie.get_buffer @buf39 : memref<64xi32, 2>
      aie.dma_bd(%buffer_36_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul26 = arith.constant 1 : i32
      aie.use_lock(@lock_35, Release, %c1_ul26)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      %c0_ul27 = arith.constant 0 : i32
      aie.use_lock(@lock_33, Acquire, %c0_ul27)
      %buffer_34_0 = aie.get_buffer @buf40 : memref<64xi32, 2>
      aie.dma_bd(%buffer_34_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul28 = arith.constant 1 : i32
      aie.use_lock(@lock_33, Release, %c1_ul28)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      %c1_ul29 = arith.constant 1 : i32
      aie.use_lock(@lock_31, Acquire, %c1_ul29)
      %buffer_32_0 = aie.get_buffer @buf41 : memref<64xi32, 2>
      aie.dma_bd(%buffer_32_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c0_ul30 = arith.constant 0 : i32
      aie.use_lock(@lock_31, Release, %c0_ul30)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %38 = aie.core(%30)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul31 = arith.constant 1 : i32
      aie.use_lock(@lock_35, Acquire, %c1_ul31)
      %c1_ul32 = arith.constant 1 : i32
      aie.use_lock(@lock_33, Acquire, %c1_ul32)
      %c0_ul33 = arith.constant 0 : i32
      aie.use_lock(@lock_31, Acquire, %c0_ul33)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %36[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %34[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %32[%arg0] : memref<64xi32, 2>
      // }
      %c1_ul34 = arith.constant 1 : i32
      aie.use_lock(@lock_31, Release, %c1_ul34)
      %c0_ul35 = arith.constant 0 : i32
      aie.use_lock(@lock_33, Release, %c0_ul35)
      %c0_ul36 = arith.constant 0 : i32
      aie.use_lock(@lock_35, Release, %c0_ul36)
      cf.br ^bb1
    }
    %39 = aie.tile(42, 2)
    %40 = aie.tile(42, 1)
    %41 = aie.tile(42, 0)
    %42 = aie.tile(0, 3)
    %43 = aie.tile(7, 5)
    aie.lock @lock_44(%43, 2)
    aie.buffer @buf38(%43) : memref<64xi32, 2>
    aie.lock @lock_46(%43, 1)
    aie.buffer @buf37(%43) : memref<64xi32, 2>
    aie.lock @lock_48(%43, 0)
    aie.buffer @buf36(%43) : memref<64xi32, 2>
    %50 = aie.mem(%43)  {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c0_ul37 = arith.constant 0 : i32
      aie.use_lock(@lock_48, Acquire, %c0_ul37)
      %buffer_49_0 = aie.get_buffer @buf36 : memref<64xi32, 2>
      aie.dma_bd(%buffer_49_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul38 = arith.constant 1 : i32
      aie.use_lock(@lock_48, Release, %c1_ul38)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      %c0_ul39 = arith.constant 0 : i32
      aie.use_lock(@lock_46, Acquire, %c0_ul39)
      %buffer_47_0 = aie.get_buffer @buf37 : memref<64xi32, 2>
      aie.dma_bd(%buffer_47_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul40 = arith.constant 1 : i32
      aie.use_lock(@lock_46, Release, %c1_ul40)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      %c1_ul41 = arith.constant 1 : i32
      aie.use_lock(@lock_44, Acquire, %c1_ul41)
      %buffer_45_0 = aie.get_buffer @buf38 : memref<64xi32, 2>
      aie.dma_bd(%buffer_45_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c0_ul42 = arith.constant 0 : i32
      aie.use_lock(@lock_44, Release, %c0_ul42)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %51 = aie.core(%43)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul43 = arith.constant 1 : i32
      aie.use_lock(@lock_48, Acquire, %c1_ul43)
      %c1_ul44 = arith.constant 1 : i32
      aie.use_lock(@lock_46, Acquire, %c1_ul44)
      %c0_ul45 = arith.constant 0 : i32
      aie.use_lock(@lock_44, Acquire, %c0_ul45)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %49[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %47[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %45[%arg0] : memref<64xi32, 2>
      // }
      %c1_ul46 = arith.constant 1 : i32
      aie.use_lock(@lock_44, Release, %c1_ul46)
      %c0_ul47 = arith.constant 0 : i32
      aie.use_lock(@lock_46, Release, %c0_ul47)
      %c0_ul48 = arith.constant 0 : i32
      aie.use_lock(@lock_48, Release, %c0_ul48)
      cf.br ^bb1
    }
    %52 = aie.tile(35, 2)
    %53 = aie.tile(35, 1)
    %54 = aie.tile(35, 0)
    %55 = aie.tile(10, 4)
    aie.lock @lock_56(%55, 2)
    aie.buffer @buf35(%55) : memref<64xi32, 2>
    aie.lock @lock_58(%55, 1)
    aie.buffer @buf34(%55) : memref<64xi32, 2>
    aie.lock @lock_60(%55, 0)
    aie.buffer @buf33(%55) : memref<64xi32, 2>
    %62 = aie.mem(%55)  {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c0_ul49 = arith.constant 0 : i32
      aie.use_lock(@lock_60, Acquire, %c0_ul49)
      %buffer_61_0 = aie.get_buffer @buf33 : memref<64xi32, 2>
      aie.dma_bd(%buffer_61_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul50 = arith.constant 1 : i32
      aie.use_lock(@lock_60, Release, %c1_ul50)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      %c0_ul51 = arith.constant 0 : i32
      aie.use_lock(@lock_58, Acquire, %c0_ul51)
      %buffer_59_0 = aie.get_buffer @buf34 : memref<64xi32, 2>
      aie.dma_bd(%buffer_59_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul52 = arith.constant 1 : i32
      aie.use_lock(@lock_58, Release, %c1_ul52)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      %c1_ul53 = arith.constant 1 : i32
      aie.use_lock(@lock_56, Acquire, %c1_ul53)
      %buffer_57_0 = aie.get_buffer @buf35 : memref<64xi32, 2>
      aie.dma_bd(%buffer_57_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c0_ul54 = arith.constant 0 : i32
      aie.use_lock(@lock_56, Release, %c0_ul54)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %63 = aie.core(%55)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul55 = arith.constant 1 : i32
      aie.use_lock(@lock_60, Acquire, %c1_ul55)
      %c1_ul56 = arith.constant 1 : i32
      aie.use_lock(@lock_58, Acquire, %c1_ul56)
      %c0_ul57 = arith.constant 0 : i32
      aie.use_lock(@lock_56, Acquire, %c0_ul57)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %61[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %59[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %57[%arg0] : memref<64xi32, 2>
      // }
      %c1_ul58 = arith.constant 1 : i32
      aie.use_lock(@lock_56, Release, %c1_ul58)
      %c0_ul59 = arith.constant 0 : i32
      aie.use_lock(@lock_58, Release, %c0_ul59)
      %c0_ul60 = arith.constant 0 : i32
      aie.use_lock(@lock_60, Release, %c0_ul60)
      cf.br ^bb1
    }
    %64 = aie.tile(34, 2)
    %65 = aie.tile(34, 1)
    %66 = aie.tile(34, 0)
    %67 = aie.tile(9, 4)
    aie.lock @lock_68(%67, 2)
    aie.buffer @buf32(%67) : memref<64xi32, 2>
    aie.lock @lock_70(%67, 1)
    aie.buffer @buf31(%67) : memref<64xi32, 2>
    aie.lock @lock_72(%67, 0)
    aie.buffer @buf30(%67) : memref<64xi32, 2>
    %74 = aie.mem(%67)  {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c0_ul61 = arith.constant 0 : i32
      aie.use_lock(@lock_72, Acquire, %c0_ul61)
      %buffer_73_0 = aie.get_buffer @buf30 : memref<64xi32, 2>
      aie.dma_bd(%buffer_73_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul62 = arith.constant 1 : i32
      aie.use_lock(@lock_72, Release, %c1_ul62)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      %c0_ul63 = arith.constant 0 : i32
      aie.use_lock(@lock_70, Acquire, %c0_ul63)
      %buffer_71_0 = aie.get_buffer @buf31 : memref<64xi32, 2>
      aie.dma_bd(%buffer_71_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul64 = arith.constant 1 : i32
      aie.use_lock(@lock_70, Release, %c1_ul64)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      %c1_ul65 = arith.constant 1 : i32
      aie.use_lock(@lock_68, Acquire, %c1_ul65)
      %buffer_69_0 = aie.get_buffer @buf32 : memref<64xi32, 2>
      aie.dma_bd(%buffer_69_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c0_ul66 = arith.constant 0 : i32
      aie.use_lock(@lock_68, Release, %c0_ul66)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %75 = aie.core(%67)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul67 = arith.constant 1 : i32
      aie.use_lock(@lock_72, Acquire, %c1_ul67)
      %c1_ul68 = arith.constant 1 : i32
      aie.use_lock(@lock_70, Acquire, %c1_ul68)
      %c0_ul69 = arith.constant 0 : i32
      aie.use_lock(@lock_68, Acquire, %c0_ul69)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %73[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %71[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %69[%arg0] : memref<64xi32, 2>
      // }
      %c1_ul70 = arith.constant 1 : i32
      aie.use_lock(@lock_68, Release, %c1_ul70)
      %c0_ul71 = arith.constant 0 : i32
      aie.use_lock(@lock_70, Release, %c0_ul71)
      %c0_ul72 = arith.constant 0 : i32
      aie.use_lock(@lock_72, Release, %c0_ul72)
      cf.br ^bb1
    }
    %76 = aie.tile(27, 2)
    %77 = aie.tile(27, 1)
    %78 = aie.tile(27, 0)
    %79 = aie.tile(1, 2)
    %80 = aie.tile(8, 4)
    aie.lock @lock_81(%80, 2)
    aie.buffer @buf29(%80) : memref<64xi32, 2>
    aie.lock @lock_83(%80, 1)
    aie.buffer @buf28(%80) : memref<64xi32, 2>
    aie.lock @lock_85(%80, 0)
    aie.buffer @buf27(%80) : memref<64xi32, 2>
    %87 = aie.mem(%80)  {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c0_ul73 = arith.constant 0 : i32
      aie.use_lock(@lock_85, Acquire, %c0_ul73)
      %buffer_86_0 = aie.get_buffer @buf27 : memref<64xi32, 2>
      aie.dma_bd(%buffer_86_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul74 = arith.constant 1 : i32
      aie.use_lock(@lock_85, Release, %c1_ul74)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      %c0_ul75 = arith.constant 0 : i32
      aie.use_lock(@lock_83, Acquire, %c0_ul75)
      %buffer_84_0 = aie.get_buffer @buf28 : memref<64xi32, 2>
      aie.dma_bd(%buffer_84_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul76 = arith.constant 1 : i32
      aie.use_lock(@lock_83, Release, %c1_ul76)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      %c1_ul77 = arith.constant 1 : i32
      aie.use_lock(@lock_81, Acquire, %c1_ul77)
      %buffer_82_0 = aie.get_buffer @buf29 : memref<64xi32, 2>
      aie.dma_bd(%buffer_82_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c0_ul78 = arith.constant 0 : i32
      aie.use_lock(@lock_81, Release, %c0_ul78)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %88 = aie.core(%80)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul79 = arith.constant 1 : i32
      aie.use_lock(@lock_85, Acquire, %c1_ul79)
      %c1_ul80 = arith.constant 1 : i32
      aie.use_lock(@lock_83, Acquire, %c1_ul80)
      %c0_ul81 = arith.constant 0 : i32
      aie.use_lock(@lock_81, Acquire, %c0_ul81)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %86[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %84[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %82[%arg0] : memref<64xi32, 2>
      // }
      %c1_ul82 = arith.constant 1 : i32
      aie.use_lock(@lock_81, Release, %c1_ul82)
      %c0_ul83 = arith.constant 0 : i32
      aie.use_lock(@lock_83, Release, %c0_ul83)
      %c0_ul84 = arith.constant 0 : i32
      aie.use_lock(@lock_85, Release, %c0_ul84)
      cf.br ^bb1
    }
    %89 = aie.tile(26, 2)
    %90 = aie.tile(26, 1)
    %91 = aie.tile(26, 0)
    %92 = aie.tile(0, 2)
    %93 = aie.tile(7, 4)
    aie.lock @lock_94(%93, 2)
    aie.buffer @buf26(%93) : memref<64xi32, 2>
    aie.lock @lock_96(%93, 1)
    aie.buffer @buf25(%93) : memref<64xi32, 2>
    aie.lock @lock_98(%93, 0)
    aie.buffer @buf24(%93) : memref<64xi32, 2>
    %100 = aie.mem(%93)  {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c0_ul85 = arith.constant 0 : i32
      aie.use_lock(@lock_98, Acquire, %c0_ul85)
      %buffer_99_0 = aie.get_buffer @buf24 : memref<64xi32, 2>
      aie.dma_bd(%buffer_99_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul86 = arith.constant 1 : i32
      aie.use_lock(@lock_98, Release, %c1_ul86)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      %c0_ul87 = arith.constant 0 : i32
      aie.use_lock(@lock_96, Acquire, %c0_ul87)
      %buffer_97_0 = aie.get_buffer @buf25 : memref<64xi32, 2>
      aie.dma_bd(%buffer_97_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul88 = arith.constant 1 : i32
      aie.use_lock(@lock_96, Release, %c1_ul88)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      %c1_ul89 = arith.constant 1 : i32
      aie.use_lock(@lock_94, Acquire, %c1_ul89)
      %buffer_95_0 = aie.get_buffer @buf26 : memref<64xi32, 2>
      aie.dma_bd(%buffer_95_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c0_ul90 = arith.constant 0 : i32
      aie.use_lock(@lock_94, Release, %c0_ul90)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %101 = aie.core(%93)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul91 = arith.constant 1 : i32
      aie.use_lock(@lock_98, Acquire, %c1_ul91)
      %c1_ul92 = arith.constant 1 : i32
      aie.use_lock(@lock_96, Acquire, %c1_ul92)
      %c0_ul93 = arith.constant 0 : i32
      aie.use_lock(@lock_94, Acquire, %c0_ul93)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %99[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %97[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %95[%arg0] : memref<64xi32, 2>
      // }
      %c1_ul94 = arith.constant 1 : i32
      aie.use_lock(@lock_94, Release, %c1_ul94)
      %c0_ul95 = arith.constant 0 : i32
      aie.use_lock(@lock_96, Release, %c0_ul95)
      %c0_ul96 = arith.constant 0 : i32
      aie.use_lock(@lock_98, Release, %c0_ul96)
      cf.br ^bb1
    }
    %102 = aie.tile(19, 2)
    %103 = aie.tile(19, 1)
    %104 = aie.tile(19, 0)
    %105 = aie.tile(10, 3)
    aie.lock @lock_106(%105, 2)
    aie.buffer @buf23(%105) : memref<64xi32, 2>
    aie.lock @lock_108(%105, 1)
    aie.buffer @buf22(%105) : memref<64xi32, 2>
    aie.lock @lock_110(%105, 0)
    aie.buffer @buf21(%105) : memref<64xi32, 2>
    %112 = aie.mem(%105)  {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c0_ul97 = arith.constant 0 : i32
      aie.use_lock(@lock_110, Acquire, %c0_ul97)
      %buffer_111_0 = aie.get_buffer @buf21 : memref<64xi32, 2>
      aie.dma_bd(%buffer_111_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul98 = arith.constant 1 : i32
      aie.use_lock(@lock_110, Release, %c1_ul98)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      %c0_ul99 = arith.constant 0 : i32
      aie.use_lock(@lock_108, Acquire, %c0_ul99)
      %buffer_109_0 = aie.get_buffer @buf22 : memref<64xi32, 2>
      aie.dma_bd(%buffer_109_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul100 = arith.constant 1 : i32
      aie.use_lock(@lock_108, Release, %c1_ul100)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      %c1_ul101 = arith.constant 1 : i32
      aie.use_lock(@lock_106, Acquire, %c1_ul101)
      %buffer_107_0 = aie.get_buffer @buf23 : memref<64xi32, 2>
      aie.dma_bd(%buffer_107_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c0_ul102 = arith.constant 0 : i32
      aie.use_lock(@lock_106, Release, %c0_ul102)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %113 = aie.core(%105)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul103 = arith.constant 1 : i32
      aie.use_lock(@lock_110, Acquire, %c1_ul103)
      %c1_ul104 = arith.constant 1 : i32
      aie.use_lock(@lock_108, Acquire, %c1_ul104)
      %c0_ul105 = arith.constant 0 : i32
      aie.use_lock(@lock_106, Acquire, %c0_ul105)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %111[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %109[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %107[%arg0] : memref<64xi32, 2>
      // }
      %c1_ul106 = arith.constant 1 : i32
      aie.use_lock(@lock_106, Release, %c1_ul106)
      %c0_ul107 = arith.constant 0 : i32
      aie.use_lock(@lock_108, Release, %c0_ul107)
      %c0_ul108 = arith.constant 0 : i32
      aie.use_lock(@lock_110, Release, %c0_ul108)
      cf.br ^bb1
    }
    %114 = aie.tile(18, 2)
    %115 = aie.tile(18, 1)
    %116 = aie.tile(18, 0)
    %117 = aie.tile(9, 3)
    aie.lock @lock_118(%117, 2)
    aie.buffer @buf20(%117) : memref<64xi32, 2>
    aie.lock @lock_120(%117, 1)
    aie.buffer @buf19(%117) : memref<64xi32, 2>
    aie.lock @lock_122(%117, 0)
    aie.buffer @buf18(%117) : memref<64xi32, 2>
    %124 = aie.mem(%117)  {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c0_ul109 = arith.constant 0 : i32
      aie.use_lock(@lock_122, Acquire, %c0_ul109)
      %buffer_123_0 = aie.get_buffer @buf18 : memref<64xi32, 2>
      aie.dma_bd(%buffer_123_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul110 = arith.constant 1 : i32
      aie.use_lock(@lock_122, Release, %c1_ul110)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      %c0_ul111 = arith.constant 0 : i32
      aie.use_lock(@lock_120, Acquire, %c0_ul111)
      %buffer_121_0 = aie.get_buffer @buf19 : memref<64xi32, 2>
      aie.dma_bd(%buffer_121_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul112 = arith.constant 1 : i32
      aie.use_lock(@lock_120, Release, %c1_ul112)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      %c1_ul113 = arith.constant 1 : i32
      aie.use_lock(@lock_118, Acquire, %c1_ul113)
      %buffer_119_0 = aie.get_buffer @buf20 : memref<64xi32, 2>
      aie.dma_bd(%buffer_119_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c0_ul114 = arith.constant 0 : i32
      aie.use_lock(@lock_118, Release, %c0_ul114)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %125 = aie.core(%117)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul115 = arith.constant 1 : i32
      aie.use_lock(@lock_122, Acquire, %c1_ul115)
      %c1_ul116 = arith.constant 1 : i32
      aie.use_lock(@lock_120, Acquire, %c1_ul116)
      %c0_ul117 = arith.constant 0 : i32
      aie.use_lock(@lock_118, Acquire, %c0_ul117)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %123[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %121[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %119[%arg0] : memref<64xi32, 2>
      // }
      %c1_ul118 = arith.constant 1 : i32
      aie.use_lock(@lock_118, Release, %c1_ul118)
      %c0_ul119 = arith.constant 0 : i32
      aie.use_lock(@lock_120, Release, %c0_ul119)
      %c0_ul120 = arith.constant 0 : i32
      aie.use_lock(@lock_122, Release, %c0_ul120)
      cf.br ^bb1
    }
    %126 = aie.tile(11, 2)
    %127 = aie.tile(11, 1)
    %128 = aie.tile(11, 0)
    %129 = aie.tile(1, 1)
    %130 = aie.tile(8, 3)
    aie.lock @lock_131(%130, 2)
    aie.buffer @buf17(%130) : memref<64xi32, 2>
    aie.lock @lock_133(%130, 1)
    aie.buffer @buf16(%130) : memref<64xi32, 2>
    aie.lock @lock_135(%130, 0)
    aie.buffer @buf15(%130) : memref<64xi32, 2>
    %137 = aie.mem(%130)  {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c0_ul121 = arith.constant 0 : i32
      aie.use_lock(@lock_135, Acquire, %c0_ul121)
      %buffer_136_0 = aie.get_buffer @buf15 : memref<64xi32, 2>
      aie.dma_bd(%buffer_136_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul122 = arith.constant 1 : i32
      aie.use_lock(@lock_135, Release, %c1_ul122)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      %c0_ul123 = arith.constant 0 : i32
      aie.use_lock(@lock_133, Acquire, %c0_ul123)
      %buffer_134_0 = aie.get_buffer @buf16 : memref<64xi32, 2>
      aie.dma_bd(%buffer_134_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul124 = arith.constant 1 : i32
      aie.use_lock(@lock_133, Release, %c1_ul124)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      %c1_ul125 = arith.constant 1 : i32
      aie.use_lock(@lock_131, Acquire, %c1_ul125)
      %buffer_132_0 = aie.get_buffer @buf17 : memref<64xi32, 2>
      aie.dma_bd(%buffer_132_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c0_ul126 = arith.constant 0 : i32
      aie.use_lock(@lock_131, Release, %c0_ul126)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %138 = aie.core(%130)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul127 = arith.constant 1 : i32
      aie.use_lock(@lock_135, Acquire, %c1_ul127)
      %c1_ul128 = arith.constant 1 : i32
      aie.use_lock(@lock_133, Acquire, %c1_ul128)
      %c0_ul129 = arith.constant 0 : i32
      aie.use_lock(@lock_131, Acquire, %c0_ul129)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %136[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %134[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %132[%arg0] : memref<64xi32, 2>
      // }
      %c1_ul130 = arith.constant 1 : i32
      aie.use_lock(@lock_131, Release, %c1_ul130)
      %c0_ul131 = arith.constant 0 : i32
      aie.use_lock(@lock_133, Release, %c0_ul131)
      %c0_ul132 = arith.constant 0 : i32
      aie.use_lock(@lock_135, Release, %c0_ul132)
      cf.br ^bb1
    }
    %139 = aie.tile(10, 1)
    %140 = aie.tile(10, 0)
    %141 = aie.tile(0, 1)
    %142 = aie.tile(7, 3)
    aie.lock @lock_143(%142, 2)
    aie.buffer @buf14(%142) : memref<64xi32, 2>
    aie.lock @lock_145(%142, 1)
    aie.buffer @buf13(%142) : memref<64xi32, 2>
    aie.lock @lock_147(%142, 0)
    aie.buffer @buf12(%142) : memref<64xi32, 2>
    %149 = aie.mem(%142)  {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c0_ul133 = arith.constant 0 : i32
      aie.use_lock(@lock_147, Acquire, %c0_ul133)
      %buffer_148_0 = aie.get_buffer @buf12 : memref<64xi32, 2>
      aie.dma_bd(%buffer_148_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul134 = arith.constant 1 : i32
      aie.use_lock(@lock_147, Release, %c1_ul134)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      %c0_ul135 = arith.constant 0 : i32
      aie.use_lock(@lock_145, Acquire, %c0_ul135)
      %buffer_146_0 = aie.get_buffer @buf13 : memref<64xi32, 2>
      aie.dma_bd(%buffer_146_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul136 = arith.constant 1 : i32
      aie.use_lock(@lock_145, Release, %c1_ul136)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      %c1_ul137 = arith.constant 1 : i32
      aie.use_lock(@lock_143, Acquire, %c1_ul137)
      %buffer_144_0 = aie.get_buffer @buf14 : memref<64xi32, 2>
      aie.dma_bd(%buffer_144_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c0_ul138 = arith.constant 0 : i32
      aie.use_lock(@lock_143, Release, %c0_ul138)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %150 = aie.core(%142)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul139 = arith.constant 1 : i32
      aie.use_lock(@lock_147, Acquire, %c1_ul139)
      %c1_ul140 = arith.constant 1 : i32
      aie.use_lock(@lock_145, Acquire, %c1_ul140)
      %c0_ul141 = arith.constant 0 : i32
      aie.use_lock(@lock_143, Acquire, %c0_ul141)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %148[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %146[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %144[%arg0] : memref<64xi32, 2>
      // }
      %c1_ul142 = arith.constant 1 : i32
      aie.use_lock(@lock_143, Release, %c1_ul142)
      %c0_ul143 = arith.constant 0 : i32
      aie.use_lock(@lock_145, Release, %c0_ul143)
      %c0_ul144 = arith.constant 0 : i32
      aie.use_lock(@lock_147, Release, %c0_ul144)
      cf.br ^bb1
    }
    %151 = aie.tile(7, 1)
    %152 = aie.tile(7, 0)
    %153 = aie.tile(10, 2)
    aie.lock @lock_154(%153, 2)
    aie.buffer @buf11(%153) : memref<64xi32, 2>
    aie.lock @lock_156(%153, 1)
    aie.buffer @buf10(%153) : memref<64xi32, 2>
    aie.lock @lock_158(%153, 0)
    aie.buffer @buf9(%153) : memref<64xi32, 2>
    %160 = aie.mem(%153)  {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c0_ul145 = arith.constant 0 : i32
      aie.use_lock(@lock_158, Acquire, %c0_ul145)
      %buffer_159_0 = aie.get_buffer @buf9 : memref<64xi32, 2>
      aie.dma_bd(%buffer_159_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul146 = arith.constant 1 : i32
      aie.use_lock(@lock_158, Release, %c1_ul146)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      %c0_ul147 = arith.constant 0 : i32
      aie.use_lock(@lock_156, Acquire, %c0_ul147)
      %buffer_157_0 = aie.get_buffer @buf10 : memref<64xi32, 2>
      aie.dma_bd(%buffer_157_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul148 = arith.constant 1 : i32
      aie.use_lock(@lock_156, Release, %c1_ul148)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      %c1_ul149 = arith.constant 1 : i32
      aie.use_lock(@lock_154, Acquire, %c1_ul149)
      %buffer_155_0 = aie.get_buffer @buf11 : memref<64xi32, 2>
      aie.dma_bd(%buffer_155_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c0_ul150 = arith.constant 0 : i32
      aie.use_lock(@lock_154, Release, %c0_ul150)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %161 = aie.core(%153)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul151 = arith.constant 1 : i32
      aie.use_lock(@lock_158, Acquire, %c1_ul151)
      %c1_ul152 = arith.constant 1 : i32
      aie.use_lock(@lock_156, Acquire, %c1_ul152)
      %c0_ul153 = arith.constant 0 : i32
      aie.use_lock(@lock_154, Acquire, %c0_ul153)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %159[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %157[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %155[%arg0] : memref<64xi32, 2>
      // }
      %c1_ul154 = arith.constant 1 : i32
      aie.use_lock(@lock_154, Release, %c1_ul154)
      %c0_ul155 = arith.constant 0 : i32
      aie.use_lock(@lock_156, Release, %c0_ul155)
      %c0_ul156 = arith.constant 0 : i32
      aie.use_lock(@lock_158, Release, %c0_ul156)
      cf.br ^bb1
    }
    %162 = aie.tile(6, 2)
    %163 = aie.tile(6, 1)
    %164 = aie.tile(6, 0)
    %165 = aie.tile(9, 2)
    aie.lock @lock_166(%165, 2)
    aie.buffer @buf8(%165) : memref<64xi32, 2>
    aie.lock @lock_168(%165, 1)
    aie.buffer @buf7(%165) : memref<64xi32, 2>
    aie.lock @lock_170(%165, 0)
    aie.buffer @buf6(%165) : memref<64xi32, 2>
    %172 = aie.mem(%165)  {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c0_ul157 = arith.constant 0 : i32
      aie.use_lock(@lock_170, Acquire, %c0_ul157)
      %buffer_171_0 = aie.get_buffer @buf6 : memref<64xi32, 2>
      aie.dma_bd(%buffer_171_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul158 = arith.constant 1 : i32
      aie.use_lock(@lock_170, Release, %c1_ul158)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      %c0_ul159 = arith.constant 0 : i32
      aie.use_lock(@lock_168, Acquire, %c0_ul159)
      %buffer_169_0 = aie.get_buffer @buf7 : memref<64xi32, 2>
      aie.dma_bd(%buffer_169_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul160 = arith.constant 1 : i32
      aie.use_lock(@lock_168, Release, %c1_ul160)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      %c1_ul161 = arith.constant 1 : i32
      aie.use_lock(@lock_166, Acquire, %c1_ul161)
      %buffer_167_0 = aie.get_buffer @buf8 : memref<64xi32, 2>
      aie.dma_bd(%buffer_167_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c0_ul162 = arith.constant 0 : i32
      aie.use_lock(@lock_166, Release, %c0_ul162)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %173 = aie.core(%165)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul163 = arith.constant 1 : i32
      aie.use_lock(@lock_170, Acquire, %c1_ul163)
      %c1_ul164 = arith.constant 1 : i32
      aie.use_lock(@lock_168, Acquire, %c1_ul164)
      %c0_ul165 = arith.constant 0 : i32
      aie.use_lock(@lock_166, Acquire, %c0_ul165)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %171[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %169[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %167[%arg0] : memref<64xi32, 2>
      // }
      %c1_ul166 = arith.constant 1 : i32
      aie.use_lock(@lock_166, Release, %c1_ul166)
      %c0_ul167 = arith.constant 0 : i32
      aie.use_lock(@lock_168, Release, %c0_ul167)
      %c0_ul168 = arith.constant 0 : i32
      aie.use_lock(@lock_170, Release, %c0_ul168)
      cf.br ^bb1
    }
    %174 = aie.tile(3, 2)
    %175 = aie.tile(3, 1)
    %176 = aie.tile(3, 0)
    %177 = aie.tile(1, 0)
    %178 = aie.tile(8, 2)
    aie.lock @lock_179(%178, 2)
    aie.buffer @buf5(%178) : memref<64xi32, 2>
    aie.lock @lock_181(%178, 1)
    aie.buffer @buf4(%178) : memref<64xi32, 2>
    aie.lock @lock_183(%178, 0)
    aie.buffer @buf3(%178) : memref<64xi32, 2>
    %185 = aie.mem(%178)  {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c0_ul169 = arith.constant 0 : i32
      aie.use_lock(@lock_183, Acquire, %c0_ul169)
      %buffer_184_0 = aie.get_buffer @buf3 : memref<64xi32, 2>
      aie.dma_bd(%buffer_184_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul170 = arith.constant 1 : i32
      aie.use_lock(@lock_183, Release, %c1_ul170)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      %c0_ul171 = arith.constant 0 : i32
      aie.use_lock(@lock_181, Acquire, %c0_ul171)
      %buffer_182_0 = aie.get_buffer @buf4 : memref<64xi32, 2>
      aie.dma_bd(%buffer_182_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul172 = arith.constant 1 : i32
      aie.use_lock(@lock_181, Release, %c1_ul172)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      %c1_ul173 = arith.constant 1 : i32
      aie.use_lock(@lock_179, Acquire, %c1_ul173)
      %buffer_180_0 = aie.get_buffer @buf5 : memref<64xi32, 2>
      aie.dma_bd(%buffer_180_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c0_ul174 = arith.constant 0 : i32
      aie.use_lock(@lock_179, Release, %c0_ul174)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %186 = aie.core(%178)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul175 = arith.constant 1 : i32
      aie.use_lock(@lock_183, Acquire, %c1_ul175)
      %c1_ul176 = arith.constant 1 : i32
      aie.use_lock(@lock_181, Acquire, %c1_ul176)
      %c0_ul177 = arith.constant 0 : i32
      aie.use_lock(@lock_179, Acquire, %c0_ul177)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %184[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %182[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %180[%arg0] : memref<64xi32, 2>
      // }
      %c1_ul178 = arith.constant 1 : i32
      aie.use_lock(@lock_179, Release, %c1_ul178)
      %c0_ul179 = arith.constant 0 : i32
      aie.use_lock(@lock_181, Release, %c0_ul179)
      %c0_ul180 = arith.constant 0 : i32
      aie.use_lock(@lock_183, Release, %c0_ul180)
      cf.br ^bb1
    }
    %187 = aie.tile(2, 2)
    %188 = aie.tile(2, 1)
    %189 = aie.tile(2, 0)
    %190 = aie.tile(0, 0)
    %191 = aie.tile(7, 2)
    aie.lock @lock_192(%191, 2)
    aie.buffer @buf2(%191) : memref<64xi32, 2>
    aie.lock @lock_194(%191, 1)
    aie.buffer @buf1(%191) : memref<64xi32, 2>
    aie.lock @lock_196(%191, 0)
    aie.buffer @buf0(%191) : memref<64xi32, 2>
    %198 = aie.mem(%191)  {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c0_ul181 = arith.constant 0 : i32
      aie.use_lock(@lock_196, Acquire, %c0_ul181)
      %buffer_197_0 = aie.get_buffer @buf0 : memref<64xi32, 2>
      aie.dma_bd(%buffer_197_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul182 = arith.constant 1 : i32
      aie.use_lock(@lock_196, Release, %c1_ul182)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      %c0_ul183 = arith.constant 0 : i32
      aie.use_lock(@lock_194, Acquire, %c0_ul183)
      %buffer_195_0 = aie.get_buffer @buf1 : memref<64xi32, 2>
      aie.dma_bd(%buffer_195_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c1_ul184 = arith.constant 1 : i32
      aie.use_lock(@lock_194, Release, %c1_ul184)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      %c1_ul185 = arith.constant 1 : i32
      aie.use_lock(@lock_192, Acquire, %c1_ul185)
      %buffer_193_0 = aie.get_buffer @buf2 : memref<64xi32, 2>
      aie.dma_bd(%buffer_193_0 : memref<64xi32, 2> offset = 0 len = 64)
      %c0_ul186 = arith.constant 0 : i32
      aie.use_lock(@lock_192, Release, %c0_ul186)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %199 = aie.core(%191)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %c1_ul187 = arith.constant 1 : i32
      aie.use_lock(@lock_196, Acquire, %c1_ul187)
      %c1_ul188 = arith.constant 1 : i32
      aie.use_lock(@lock_194, Acquire, %c1_ul188)
      %c0_ul189 = arith.constant 0 : i32
      aie.use_lock(@lock_192, Acquire, %c0_ul189)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %197[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %195[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %193[%arg0] : memref<64xi32, 2>
      // }
      %c1_ul190 = arith.constant 1 : i32
      aie.use_lock(@lock_192, Release, %c1_ul190)
      %c0_ul191 = arith.constant 0 : i32
      aie.use_lock(@lock_194, Release, %c0_ul191)
      %c0_ul192 = arith.constant 0 : i32
      aie.use_lock(@lock_196, Release, %c0_ul192)
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
