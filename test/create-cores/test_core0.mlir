//===- test_core0.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-cores %s | FileCheck %s

// CHECK-LABEL: module @test_core0 {
// CHECK:         %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:         %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) : memref<256xi32>
// CHECK:         %[[VAL_2:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:           aie.end
// CHECK:         }
// CHECK:         %[[VAL_3:.*]] = memref.alloc() : memref<256xi32>
// CHECK:         func.func @host_task() {
// CHECK:           return
// CHECK:         }
// CHECK:         %[[VAL_4:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:           aie.end
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
 aie.device(xcvc1902) {
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
