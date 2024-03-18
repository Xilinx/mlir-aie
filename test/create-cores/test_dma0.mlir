//===- test_dma0.mlir ------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-cores --aie-lower-memcpy %s | FileCheck %s

// CHECK-LABEL: module @test_dma0 {
// CHECK:         %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:         %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) : memref<256xi32>
// CHECK:         %[[VAL_2:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:           %[[VAL_3:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           aiex.useToken @token0(Acquire, 1)
// CHECK:           aie.dma_bd(%[[VAL_1]] : memref<256xi32>) {len = 256 : i32}
// CHECK:           aiex.useToken @token0(Release, 2)
// CHECK:           aie.next_bd ^bb2
// CHECK:         ^bb2:
// CHECK:           aie.end
// CHECK:         }
// CHECK:         %[[VAL_4:.*]] = aie.tile(2, 2)
// CHECK:         %[[VAL_5:.*]] = aie.buffer(%[[VAL_4]]) : memref<256xi32>
// CHECK:         %[[VAL_6:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:           %[[VAL_7:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           aiex.useToken @token0(Acquire, 1)
// CHECK:           aie.dma_bd(%[[VAL_5]] : memref<256xi32>) {len = 256 : i32}
// CHECK:           aiex.useToken @token0(Release, 2)
// CHECK:           aie.next_bd ^bb2
// CHECK:         ^bb2:
// CHECK:           aie.end
// CHECK:         }
// CHECK:         %[[VAL_8:.*]] = memref.alloc() : memref<256xi32>
// CHECK:         %[[VAL_9:.*]] = memref.alloc() : memref<256xi32>
// CHECK:         aiex.token(0) {sym_name = "token0"}
// CHECK:         %[[VAL_10:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:           aiex.useToken @token0(Acquire, 0)
// CHECK:           aiex.useToken @token0(Release, 1)
// CHECK:           aie.end
// CHECK:         }
// CHECK:         aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:         %[[VAL_11:.*]] = aie.core(%[[VAL_4]]) {
// CHECK:           aiex.useToken @token0(Acquire, 2)
// CHECK:           aiex.useToken @token0(Release, 3)
// CHECK:           aie.end
// CHECK:         }
// CHECK:       }

// Lowering Std::FuncOp and Std::CallOp with (aie.x, aie.y) attributes to AIE::CoreOp,
// AIE::MemOp, and AIE::TileOp
// Lowering AIE::memcpy to AIE::DMAStartOp and AIE::DMABDOp
// memcpy is a logical op represents memory transfer of buffers from one core's memory region to
// a different core's memory region
// When lowering, we need to insert the DMA ops on both the sender and receiver
// In arith.addition, we want to model the block descriptions for DMA transfers.
// In the AIE array device, each DMA (of a Core Tile) has 16 Block Descriptions (BD) that are shared
// among four DMA channels (MM2S0, MM2S1, S2MM0, S2MM1). The BDs can be chained together so that
// a DMA channel can process one transfer after another
//
// For now, an MLIR Block is used to denote one BD, and branch op is used to denote the relationship
// of the BDs (what would be the next BD, etc.)
// Using Blocks also allows us to inject lock/token Ops more naturally (instead of having to create
// a new op with locking mechanism -- which is clunky and makes it harder to do locking analysis ...)
module @test_dma0 {
 aie.device(xcvc1902) {
  %t11 = aie.tile(1, 1)
  %t22 = aie.tile(2, 2)

  %buf0 = memref.alloc() : memref<256xi32>
  
  %buf1 = memref.alloc() : memref<256xi32>

  aiex.token(0) { sym_name="token0" }

  func.func @task0(%arg0: memref<256xi32>) -> () {
    aiex.useToken @token0(Acquire, 0)
    // code
    aiex.useToken @token0(Release, 1)
    return
  }

  func.func @task1(%arg0: memref<256xi32>) -> () {
    aiex.useToken @token0(Acquire, 2)
    // code
    aiex.useToken @token0(Release, 3)
    return
  }

  func.call @task0(%buf0) { aie.x = 1, aie.y = 1 } : (memref<256xi32>) -> ()
  aiex.memcpy @token0(1, 2) (%t11 : <%buf0, 0, 256>, %t22 : <%buf1, 0, 256>) : (memref<256xi32>, memref<256xi32>)
  func.call @task1(%buf1) { aie.x = 2, aie.y = 2 } : (memref<256xi32>) -> ()
 }
}
