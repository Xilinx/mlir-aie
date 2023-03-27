//===- test_core0.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-cores %s | FileCheck %s

// CHECK-LABEL: module @test_core0 {
// CHECK:         %[[VAL_0:.*]] = AIE.tile(1, 1)
// CHECK:         %[[VAL_1:.*]] = AIE.buffer(%[[VAL_0]]) : memref<256xi32>
// CHECK:         %[[VAL_2:.*]] = AIE.mem(%[[VAL_0]]) {
// CHECK:           AIE.end
// CHECK:         }
// CHECK:         %[[VAL_3:.*]] = memref.alloc() : memref<256xi32>
// CHECK:         func.func @host_task() {
// CHECK:           return
// CHECK:         }
// CHECK:         %[[VAL_4:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:           AIE.end
// CHECK:         }
// CHECK:         func.call @host_task() : () -> ()
// CHECK:       }

// Lowering Std::FuncOp and Std::CallOp with (aie.x, aie.y) attributes to AIE::CoreOp,
// AIE::MemOp, and AIE::TileOp
// Basic test
// Things to do when lowering:
//   - create tile, core, and mem instances if they do not exist already
//   - convert call operands to AIE::BufferOp in the top-level module
//   - clone function body into core's region; map the function arguments to the
//     corresponding newly created buffer ops
module @test_core0 {
 AIE.device(xcvc1902) {
  %buf = memref.alloc() : memref<256xi32>

  func.func @aie_task(%arg0: memref<256xi32>) -> () {
    return
  }

  func.func @host_task() -> () {
    return
  }

  func.call @aie_task(%buf) { aie.x = 1, aie.y = 1 } : (memref<256xi32>) -> ()
  func.call @host_task() : () -> ()
 }
}
