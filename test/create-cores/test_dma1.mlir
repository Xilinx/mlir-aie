//===- test_dma1.mlir ------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-cores --aie-lower-memcpy %s | FileCheck %s
// XFAIL: *

// FIXCore : DMA ops have issues here: reuse of DMAs is bad, also should chain from one DMA op to the next.

// CHECK-LABEL: module @test_dma1 {
// CHECK:         %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:         %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) : memref<256xi32>
// CHECK:         %[[VAL_2:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:           %[[VAL_3:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:         ^bb1:
// CHECK:           aiex.useToken @token0(Acquire, 1)
// CHECK:           aie.dma_bd(%[[VAL_1]] : memref<256xi32>) {len = 256 : i32}
// CHECK:           aiex.useToken @token0(Release, 2)
// CHECK:           aie.next_bd ^bb4
// CHECK:         ^bb2:
// CHECK:           %[[VAL_4:.*]] = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
// CHECK:         ^bb3:
// CHECK:           aiex.useToken @token1(Acquire, 1)
// CHECK:           aie.dma_bd(%[[VAL_1]] : memref<256xi32>) {len = 256 : i32}
// CHECK:           aiex.useToken @token1(Release, 2)
// CHECK:           aie.next_bd ^bb4
// CHECK:         ^bb4:
// CHECK:           aie.end
// CHECK:         }
// CHECK:         %[[VAL_5:.*]] = aie.tile(2, 2)
// CHECK:         %[[VAL_6:.*]] = aie.buffer(%[[VAL_5]]) : memref<256xi32>
// CHECK:         %[[VAL_7:.*]] = aie.mem(%[[VAL_5]]) {
// CHECK:           %[[VAL_8:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           aiex.useToken @token0(Acquire, 1)
// CHECK:           aie.dma_bd(%[[VAL_6]] : memref<256xi32>) {len = 256 : i32}
// CHECK:           aiex.useToken @token0(Release, 2)
// CHECK:           aie.next_bd ^bb2
// CHECK:         ^bb2:
// CHECK:           aie.end
// CHECK:         }
// CHECK:         %[[VAL_9:.*]] = aie.tile(3, 3)
// CHECK:         %[[VAL_10:.*]] = aie.buffer(%[[VAL_9]]) : memref<256xi32>
// CHECK:         %[[VAL_11:.*]] = aie.mem(%[[VAL_9]]) {
// CHECK:           %[[VAL_12:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           aiex.useToken @token1(Acquire, 1)
// CHECK:           aie.dma_bd(%[[VAL_10]] : memref<256xi32>) {len = 256 : i32}
// CHECK:           aiex.useToken @token1(Release, 2)
// CHECK:           aie.next_bd ^bb2
// CHECK:         ^bb2:
// CHECK:           aie.end
// CHECK:         }
// CHECK:         %[[VAL_13:.*]] = memref.alloc() : memref<256xi32>
// CHECK:         %[[VAL_14:.*]] = memref.alloc() : memref<256xi32>
// CHECK:         %[[VAL_15:.*]] = memref.alloc() : memref<256xi32>
// CHECK:         aiex.token(0) {sym_name = "token0"}
// CHECK:         aiex.token(0) {sym_name = "token1"}
// CHECK:         %[[VAL_16:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:           aiex.useToken @token0(Acquire, 0)
// CHECK:           aiex.useToken @token1(Acquire, 0)
// CHECK:           aiex.useToken @token0(Release, 1)
// CHECK:           aiex.useToken @token1(Release, 1)
// CHECK:           aie.end
// CHECK:         }
// CHECK:         aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_5]], DMA : 0)
// CHECK:         aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_9]], DMA : 0)
// CHECK:         %[[VAL_17:.*]] = aie.core(%[[VAL_5]]) {
// CHECK:           aiex.useToken @token0(Acquire, 2)
// CHECK:           aiex.useToken @token0(Release, 3)
// CHECK:           aie.end
// CHECK:         }
// CHECK:         %[[VAL_18:.*]] = aie.core(%[[VAL_9]]) {
// CHECK:           aiex.useToken @token1(Acquire, 2)
// CHECK:           aiex.useToken @token1(Release, 3)
// CHECK:           aie.end
// CHECK:         }
// CHECK:       }

// Lowering Std::FuncOp and Std::CallOp with (aie.x, aie.y) attributes to AIE::CoreOp,
// AIE::MemOp, and AIE::TileOp
// Lowering AIE::memcpy to AIE::DMAStartOp and AIE::DMABDOp
// single producer, multiple consumers
module @test_dma1 {
 aie.device(xcvc1902) {
  %t11 = aie.tile(1, 1) // producer
  %t22 = aie.tile(2, 2) // consumer
  %t33 = aie.tile(3, 3) // consumer

  %buf0 = memref.alloc() : memref<256xi32>
  %buf1 = memref.alloc() : memref<256xi32>
  %buf2 = memref.alloc() : memref<256xi32>

  aiex.token(0) { sym_name="token0" }
  aiex.token(0) { sym_name="token1" }

  func.func @task0(%arg0: memref<256xi32>) -> () {
    aiex.useToken @token0(Acquire, 0)
    aiex.useToken @token1(Acquire, 0)
    // code
    aiex.useToken @token0(Release, 1)
    aiex.useToken @token1(Release, 1)
    return
  }

  func.func @task1(%arg0: memref<256xi32>) -> () {
    aiex.useToken @token0(Acquire, 2)
    // code
    aiex.useToken @token0(Release, 3)
    return
  }

  func.func @task2(%arg0: memref<256xi32>) -> () {
    aiex.useToken @token1(Acquire, 2)
    // code
    aiex.useToken @token1(Release, 3)
    return
  }

  func.call @task0(%buf0) { aie.x = 1, aie.y = 1 } : (memref<256xi32>) -> ()
  aiex.memcpy @token0(1, 2) (%t11 : <%buf0, 0, 256>, %t22 : <%buf1, 0, 256>) : (memref<256xi32>, memref<256xi32>)
  aiex.memcpy @token1(1, 2) (%t11 : <%buf0, 0, 256>, %t33 : <%buf2, 0, 256>) : (memref<256xi32>, memref<256xi32>)
  func.call @task1(%buf1) { aie.x = 2, aie.y = 2 } : (memref<256xi32>) -> ()
  func.call @task2(%buf2) { aie.x = 3, aie.y = 3 } : (memref<256xi32>) -> ()
 }
}
