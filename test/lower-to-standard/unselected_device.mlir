//===- unselected_device.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ops this pass must lower are only lowered in the selected device; an
// unselected device is erased with its contents.

// RUN: aie-opt --aie-standard-lowering="device=a" %s | FileCheck %s

// CHECK: module attributes {llvm.target_triple = "aie2"}
// CHECK: func.func @core_0_2()
// CHECK-NOT: aie.
module {
  aie.device(npu1) @a {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) {
      aie.end
    }
  }
  aie.device(xcve3858) @b {
    %t = aie.tile(2, 3)
    %l = aie.lock(%t, 0)
    %buf = aie.buffer(%t) {sym_name = "b"} : memref<16xi32>
    %c = aie.core(%t) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l, AcquireGreaterEqual, %c1)
      %ch = arith.constant 0 : i32
      aie.put_stream(%ch : i32, %ch : i32)
      aie.end
    }
  }
}
