//===- example1.mlir -------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// This is the logical netlist of mapping code to AIE array. This is closer to the application-level
// Tokens (logical locks) are used to denote the execution order of the tiles
// Attributes are used to specify binding code to a tile
// Memcpy op is used to perform logical memory transfer from one tile to another
//
// In the example below, tile(3, 3) is a producer, while tile(4, 2) and tile(4, 4) are consumers.
// Since these tiles cannot share memory physically (they are not abut), a memory transfer op is used
// to explicitly move buffers from the producer to the consumers

// CHECK-LABEL: module @example1 {
// CHECK:       }

module @example1 {
  %t33 = AIE.tile(3, 3)
  %t42 = AIE.tile(4, 2)
  %t44 = AIE.tile(4, 4)

  %buf0 = memref.alloc() : memref<256xi32>
  %buf1 = memref.alloc() : memref<256xi32>
  %buf2 = memref.alloc() : memref<256xi32>

  AIE.token(0) { sym_name="token0" }
  AIE.token(0) { sym_name="token1" }

  func.func @task0(%arg0: memref<256xi32>, %arg1: i32) -> () {
    AIE.useToken @token0(Acquire, 0)
    AIE.useToken @token1(Acquire, 0)

    // code
    %i = arith.constant 8: index
    //%k = arith.constant 1: i32
    memref.store %arg1, %arg0[%i] : memref<256xi32>
    //store %k, %arg0[%i] : memref<256xi32>

    AIE.useToken @token0(Release, 1)
    AIE.useToken @token1(Release, 1)
    return
  }

  func.func @task1(%arg0: memref<256xi32>) -> () {
    AIE.useToken @token0(Acquire, 2)

    // code

    AIE.useToken @token0(Release, 3)
    return
  }

  func.func @task2(%arg0: memref<256xi32>) -> () {
    AIE.useToken @token1(Acquire, 2)

    // code

    AIE.useToken @token1(Release, 3)
    return
  }

  func.func @task3() -> () {
    return
  }

  %t0 = arith.constant 19 : i32
  func.call @task0(%buf0, %t0) { aie.x = 3, aie.y = 3 } : (memref<256xi32>, i32) -> ()
  func.call @task1(%buf1) { aie.x = 4, aie.y = 2 } : (memref<256xi32>) -> ()
  func.call @task2(%buf2) { aie.x = 4, aie.y = 4 } : (memref<256xi32>) -> ()
  func.call @task3() : () -> ()

  AIE.memcpy @token0(1, 2) (%t33 : <%buf0, 0, 256>, %t42 : <%buf1, 0, 256>) : (memref<256xi32>, memref<256xi32>)
  AIE.memcpy @token1(1, 2) (%t33 : <%buf0, 0, 256>, %t44 : <%buf2, 0, 256>) : (memref<256xi32>, memref<256xi32>)
}
