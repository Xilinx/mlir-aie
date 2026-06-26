//===- core_uses_device_constant.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering %s | FileCheck %s

// Regression test: a core may reference a constant defined in the parent device
// body (the constant folder hoists constants out of non-IsolatedFromAbove ops
// like aie.runtime_sequence, and CSE then merges them with a core's own
// constants). When the core is outlined into a func and the device is erased,
// such a device-level constant would still have a use from the core's body,
// crashing device removal. The constant must instead be cloned into the core
// function so it is self-contained.

// The hoisted constant is cloned into the core function, not left at the device
// level.
// CHECK-LABEL: func.func @core_0_2()
// CHECK: %[[C:.*]] = arith.constant 7 : i32
// CHECK: arith.addi %{{.*}}, %[[C]] : i32

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<8xi32>
    // Constant defined in the device body but used inside the core.
    %dc = arith.constant 7 : i32
    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %v = memref.load %b[%c0] : memref<8xi32>
      %a = arith.addi %v, %dc : i32
      memref.store %a, %b[%c0] : memref<8xi32>
      aie.end
    }
  }
}
