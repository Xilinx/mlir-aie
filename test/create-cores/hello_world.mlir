//===- hello_world.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-cores --aie-lower-memcpy %s | FileCheck %s

// CHECK-LABEL: module @hello_world {
// CHECK:         %[[VAL_0:.*]] = AIE.tile(3, 3)
// CHECK:         %[[VAL_1:.*]] = AIE.buffer(%[[VAL_0]]) : memref<512xi32>
// CHECK:         %[[VAL_2:.*]] = AIE.mem(%[[VAL_0]]) {
// CHECK:           %[[VAL_3:.*]] = AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           AIEX.useToken @token0(Acquire, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_1]] : memref<512xi32>, 0, 512>, 0)
// CHECK:           AIEX.useToken @token0(Release, 2)
// CHECK:           AIE.nextBd ^bb2
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         }
// CHECK:         %[[VAL_4:.*]] = AIE.tile(4, 4)
// CHECK:         %[[VAL_5:.*]] = AIE.buffer(%[[VAL_4]]) : memref<512xi32>
// CHECK:         %[[VAL_6:.*]] = AIE.mem(%[[VAL_4]]) {
// CHECK:           %[[VAL_7:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           AIEX.useToken @token0(Acquire, 1)
// CHECK:           AIE.dmaBd(<%[[VAL_5]] : memref<512xi32>, 0, 512>, 0)
// CHECK:           AIEX.useToken @token0(Release, 2)
// CHECK:           AIE.nextBd ^bb2
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         }
// CHECK:         %[[VAL_8:.*]] = memref.alloc() : memref<512xi32>
// CHECK:         %[[VAL_9:.*]] = memref.alloc() : memref<512xi32>
// CHECK:         AIEX.token(0) {sym_name = "token0"}
// CHECK:         %[[VAL_10:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:           AIEX.useToken @token0(Acquire, 0)
// CHECK:           %[[VAL_11:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : i32
// CHECK:           memref.store %[[VAL_12]], %[[VAL_1]]{{\[}}%[[VAL_11]]] : memref<512xi32>
// CHECK:           AIEX.useToken @token0(Release, 1)
// CHECK:           AIE.end
// CHECK:         }
// CHECK:         %[[VAL_13:.*]] = AIE.core(%[[VAL_4]]) {
// CHECK:           AIEX.useToken @token0(Acquire, 2)
// CHECK:           %[[VAL_14:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_15:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_14]]] : memref<512xi32>
// CHECK:           AIEX.useToken @token0(Release, 3)
// CHECK:           AIE.end
// CHECK:         }
// CHECK:         AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_4]], DMA : 0)
// CHECK:       }

module @hello_world {
 AIE.device(xcvc1902) {

  %tile33 = AIE.tile(3, 3)
  %tile44 = AIE.tile(4, 4)

  %buf0 = memref.alloc() : memref<512xi32>
  %buf1 = memref.alloc() : memref<512xi32>

  AIEX.token(0) { sym_name="token0" }

  func.func @producer(%arg0: memref<512xi32>) -> () {
    AIEX.useToken @token0(Acquire, 0)
    %i = arith.constant 16 : index
    %val = arith.constant 1 : i32
    memref.store %val, %arg0[%i] : memref<512xi32>
    AIEX.useToken @token0(Release, 1)
    return
  }

  func.func @consumer(%arg0: memref<512xi32>) -> () {
    AIEX.useToken @token0(Acquire, 2)
    %i = arith.constant 16 : index
    %val = memref.load %arg0[%i] : memref<512xi32>
    AIEX.useToken @token0(Release, 3)
    return
  }

  func.call @producer(%buf0) { aie.x = 3, aie.y = 3 } : (memref<512xi32>) -> () // write 1 to buf[16]
  func.call @consumer(%buf1) { aie.x = 4, aie.y = 4 } : (memref<512xi32>) -> () // read buf[16]

  AIEX.memcpy @token0(1, 2) (%tile33 : <%buf0, 0, 512>, %tile44 : <%buf1, 0, 512>) : (memref<512xi32>, memref<512xi32>)
 }
}
