//===- core_uses_device_constant.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --canonicalize --cse %s | FileCheck %s

// Regression test: an aie.core and an aie.runtime_sequence may each hold an
// identical constant. aie.runtime_sequence is not IsolatedFromAbove, so without
// AIEDialectFoldInterface::shouldMaterializeInto claiming it, the folder would
// hoist its constant up to the IsolatedFromAbove aie.device body, where CSE
// merges it with the core's identical constant. The core would then reference a
// device-level value that is erased when the core is outlined by
// --aie-standard-lowering, crashing device removal. Materializing constants
// into the core and the sequence keeps them in sibling regions so CSE cannot
// merge them and the core stays self-contained.

// No constant is hoisted to the device body; each region keeps its own.
// CHECK-LABEL: aie.device
// CHECK-NOT:     arith.constant
// CHECK:       aie.core
// CHECK:         arith.constant 1 : i32
// CHECK:       aie.runtime_sequence
// CHECK:         arith.constant 1 : i32

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<8xi32>
    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : i32
      %v = memref.load %b[%c0] : memref<8xi32>
      %a = arith.addi %v, %c1 : i32
      memref.store %a, %b[%c0] : memref<8xi32>
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<8xi32>) {
      %addr = arith.constant 119300 : i32
      %v1 = arith.constant 1 : i32
      aiex.npu.write32(%addr, %v1) : i32, i32
    }
  }
}
