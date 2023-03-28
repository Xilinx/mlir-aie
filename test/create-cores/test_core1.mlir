//===- test_core1.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-cores %s | FileCheck %s

// CHECK-LABEL: module @test_core1 {
// CHECK:         %[[VAL_0:.*]] = AIE.tile(1, 1)
// CHECK:         %[[VAL_1:.*]] = AIE.buffer(%[[VAL_0]]) : memref<256xi32>
// CHECK:         %[[VAL_2:.*]] = AIE.buffer(%[[VAL_0]]) : memref<1xi32>
// CHECK:         %[[VAL_3:.*]] = AIE.mem(%[[VAL_0]]) {
// CHECK:           AIE.end
// CHECK:         }
// CHECK:         %[[VAL_4:.*]] = memref.alloc() : memref<256xi32>
// CHECK:         func.func @host_task() {
// CHECK:           return
// CHECK:         }
// CHECK:         %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:         %[[VAL_6:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_7]]] : memref<1xi32>
// CHECK:           %[[VAL_9:.*]] = arith.constant 10 : index
// CHECK:           memref.store %[[VAL_8]], %[[VAL_1]]{{\[}}%[[VAL_9]]] : memref<256xi32>
// CHECK:           AIE.end
// CHECK:         }
// CHECK:         func.call @host_task() : () -> ()
// CHECK:       }

// Lowering Std::FuncOp and Std::CallOp with (aie.x, aie.y) attributes to AIE::CoreOp,
// AIE::MemOp, and AIE::TileOp
// In this test, the aie func have both memref argument and scalar argument
// We promote the scalar argument to memref kind (single-element)
// For now, we only support scalar type of int type or float type
module @test_core1 {
 AIE.device(xcvc1902) {
  %buf = memref.alloc() : memref<256xi32>

  func.func @aie_task(%arg0: memref<256xi32>, %arg1: i32) -> () {
    %i = arith.constant 10 : index
    memref.store %arg1, %arg0[%i] : memref<256xi32>
    return
  }

  func.func @host_task() -> () {
    return
  }

  %a = arith.constant 0 : i32
  func.call @aie_task(%buf, %a) { aie.x = 1, aie.y = 1 } : (memref<256xi32>, i32) -> ()
  func.call @host_task() : () -> ()
 }
}
