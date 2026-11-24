//===- unselected_core.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Exporting one core ignores ops that would fail to lower in another core;
// that core is erased.

// RUN: aie-opt --aie-standard-lowering="tilecol=0 tilerow=2" %s | FileCheck %s
// RUN: not aie-opt --aie-standard-lowering="tilecol=0 tilerow=3" %s 2>&1 | FileCheck %s --check-prefix=SELECTED

// CHECK: func.func @core_0_2()
// CHECK-NOT: core_0_3
// CHECK-NOT: aie.put_stream

// SELECTED: failed to legalize operation 'aie.put_stream'

module {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %c02 = aie.core(%t02) {
      aie.end
    }
    %c03 = aie.core(%t03) {
      %ch = arith.constant 0 : i32
      %v = arith.constant 0 : i128
      aie.put_stream(%ch : i32, %v : i128)
      aie.end
    }
  }
}
