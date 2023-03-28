//===- test_dma2.mlir ------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: stephenn
// RUN: aie-opt --aie-create-cores %s | FileCheck %s

// CHECK-LABEL: module @test_dma2 {
// CHECK:         %[[VAL_0:.*]] = AIE.tile(1, 1)
// CHECK:         %[[VAL_1:.*]] = AIE.buffer(%[[VAL_0]]) : memref<256xi32>
// CHECK:         %[[VAL_2:.*]] = AIE.mem(%[[VAL_0]]) {
// CHECK:           %[[VAL_3:.*]] = AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           AIEX.useToken @token0(Acquire, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_1]] : memref<256xi32>, 0, 256>, 0)
// CHECK:           AIEX.useToken @token0(Release, 2)
// CHECK:           AIE.nextBd ^bb2
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         }
// CHECK:         %[[VAL_4:.*]] = AIE.tile(2, 2)
// CHECK:         %[[VAL_5:.*]] = AIE.buffer(%[[VAL_4]]) : memref<256xi32>
// CHECK:         %[[VAL_6:.*]] = AIE.mem(%[[VAL_4]]) {
// CHECK:           %[[VAL_7:.*]] = AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           AIEX.useToken @token1(Acquire, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_5]] : memref<256xi32>, 0, 256>, 0)
// CHECK:           AIEX.useToken @token1(Release, 2)
// CHECK:           AIE.nextBd ^bb2
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         }
// CHECK:         %[[VAL_8:.*]] = AIE.tile(3, 3)
// CHECK:         %[[VAL_9:.*]] = AIE.buffer(%[[VAL_8]]) : memref<256xi32>
// CHECK:         %[[VAL_10:.*]] = AIE.buffer(%[[VAL_8]]) : memref<256xi32>
// CHECK:         %[[VAL_11:.*]] = AIE.mem(%[[VAL_8]]) {
// CHECK:           %[[VAL_12:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:         ^bb1:
// CHECK:           AIEX.useToken @token0(Acquire, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_9]] : memref<256xi32>, 0, 256>, 0)
// CHECK:           AIEX.useToken @token0(Release, 2)
// CHECK:           AIE.nextBd ^bb4
// CHECK:         ^bb2:
// CHECK:           %[[VAL_13:.*]] = AIE.dmaStart(S2MM, 0, ^bb3, ^bb4)
// CHECK:         ^bb3:
// CHECK:           AIEX.useToken @token1(Acquire, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_10]] : memref<256xi32>, 0, 256>, 0)
// CHECK:           AIEX.useToken @token1(Release, 2)
// CHECK:           AIE.nextBd ^bb4
// CHECK:         ^bb4:
// CHECK:           AIE.end
// CHECK:         }
// CHECK:         %[[VAL_14:.*]] = memref.alloc() : memref<256xi32>
// CHECK:         %[[VAL_15:.*]] = memref.alloc() : memref<256xi32>
// CHECK:         %[[VAL_16:.*]] = memref.alloc() : memref<256xi32>
// CHECK:         %[[VAL_17:.*]] = memref.alloc() : memref<256xi32>
// CHECK:         AIEX.token(0) {sym_name = "token0"}
// CHECK:         AIEX.token(0) {sym_name = "token1"}
// CHECK:         %[[VAL_18:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:           AIEX.useToken @token0(Acquire, 0)
// CHECK:           AIEX.useToken @token0(Release, 1)
// CHECK:           AIE.end
// CHECK:         }
// CHECK:         %[[VAL_19:.*]] = AIE.core(%[[VAL_4]]) {
// CHECK:           AIEX.useToken @token1(Acquire, 0)
// CHECK:           AIEX.useToken @token1(Release, 1)
// CHECK:           AIE.end
// CHECK:         }
// CHECK:         AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_8]], DMA : 0)
// CHECK:         AIE.flow(%[[VAL_4]], DMA : 0, %[[VAL_8]], DMA : 1)
// CHECK:         %[[VAL_20:.*]] = AIE.core(%[[VAL_8]]) {
// CHECK:           AIEX.useToken @token0(Acquire, 2)
// CHECK:           AIEX.useToken @token1(Acquire, 2)
// CHECK:           AIEX.useToken @token0(Release, 3)
// CHECK:           AIEX.useToken @token1(Release, 3)
// CHECK:           AIE.end
// CHECK:         }
// CHECK:       }

// Lowering Std::FuncOp and Std::CallOp with (aie.x, aie.y) attributes to AIE::CoreOp,
// AIE::MemOp, and AIE::TileOp
// Lowering Std::FuncOp and Std::CallOp with (aie.x, aie.y) attributes to AIE::CoreModuleOp
// Lowering AIE::memcpy to AIE::DMAStartOp and AIE::DMABDOp
// multiple producers, single consumer
module @test_dma2 {
 AIE.device(xcvc1902) {
  %t11 = AIE.tile(1, 1) // producer
  %t22 = AIE.tile(2, 2) // producer
  %t33 = AIE.tile(3, 3) // consumer

  %buf0 = memref.alloc() : memref<256xi32>
  %buf1 = memref.alloc() : memref<256xi32>
  %buf2 = memref.alloc() : memref<256xi32>
  %buf3 = memref.alloc() : memref<256xi32>

  AIEX.token(0) { sym_name="token0" }
  AIEX.token(0) { sym_name="token1" }

  func.func @task0(%arg0: memref<256xi32>) -> () {
    AIEX.useToken @token0(Acquire, 0)
    // code
    AIEX.useToken @token0(Release, 1)
    return
  }

  func.func @task1(%arg0: memref<256xi32>) -> () {
    AIEX.useToken @token1(Acquire, 0)
    // code
    AIEX.useToken @token1(Release, 1)
    return
  }

  func.func @task2(%arg0: memref<256xi32>, %arg1: memref<256xi32>) -> () {
    AIEX.useToken @token0(Acquire, 2)
    AIEX.useToken @token1(Acquire, 2)
    // code
    AIEX.useToken @token0(Release, 3)
    AIEX.useToken @token1(Release, 3)
    return
  }

  func.call @task0(%buf0) { aie.x = 1, aie.y = 1 } : (memref<256xi32>) -> ()
  func.call @task1(%buf1) { aie.x = 2, aie.y = 2 } : (memref<256xi32>) -> ()
  AIEX.memcpy @token0(1, 2) (%t11 : <%buf0, 0, 256>, %t33 : <%buf2, 0, 256>) : (memref<256xi32>, memref<256xi32>)
  AIEX.memcpy @token1(1, 2) (%t22 : <%buf1, 0, 256>, %t33 : <%buf3, 0, 256>) : (memref<256xi32>, memref<256xi32>)
  func.call @task2(%buf2, %buf3) { aie.x = 3, aie.y = 3 } : (memref<256xi32>, memref<256xi32>) -> ()
 }
}
