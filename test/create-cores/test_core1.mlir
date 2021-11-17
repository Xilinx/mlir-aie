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
// CHECK-NEXT:    %0 = AIE.tile(1, 1)
// CHECK-NEXT:    %1 = AIE.buffer(%0) : memref<256xi32>
// CHECK-NEXT:    %2 = AIE.buffer(%0) : memref<1xi32>
// CHECK-NEXT:    %3 = AIE.mem(%0) {
// CHECK-NEXT:        AIE.end
// CHECK-NEXT:    }
// CHECK-NEXT:    %4 = memref.alloc() : memref<256xi32>
// CHECK-NEXT:    func @host_task() {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %5 = AIE.core(%0) {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %6 = memref.load %2[%c0] : memref<1xi32>
// CHECK-NEXT:      %c10 = arith.constant 10 : index
// CHECK-NEXT:      memref.store %6, %1[%c10] : memref<256xi32>
// CHECK-NEXT:      AIE.end
// CHECK-NEXT:    }
// CHECK-NEXT:    call @host_task() : () -> ()
// CHECK-NEXT:  }

// Lowering Std::FuncOp and Std::CallOp with (aie.x, aie.y) attributes to AIE::CoreOp,
// AIE::MemOp, and AIE::TileOp
// In this test, the aie func have both memref argument and scalar argument
// We promote the scalar argument to memref kind (single-element)
// For now, we only support scalar type of int type or float type
module @test_core1 {
  %buf = memref.alloc() : memref<256xi32>

  func @aie_task(%arg0: memref<256xi32>, %arg1: i32) -> () {
    %i = arith.constant 10 : index
    memref.store %arg1, %arg0[%i] : memref<256xi32>
    return
  }

  func @host_task() -> () {
    return
  }

  %a = arith.constant 0 : i32
  call @aie_task(%buf, %a) { aie.x = 1, aie.y = 1 } : (memref<256xi32>, i32) -> ()
  call @host_task() : () -> ()
}
