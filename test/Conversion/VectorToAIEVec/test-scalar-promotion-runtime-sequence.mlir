//===- test-scalar-promotion-runtime-sequence.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Regression for https://github.com/Xilinx/mlir-aie/issues/3549.
//
// LowerVectorToAIEVec promotes scalar arith.minsi/maxsi/shrsi to vector
// aievec ops (see test-scalar-promotion.mlir) to work around an AIE2 backend
// crash when lowering aie.core kernel bodies. `aie.runtime_sequence` bodies
// are host-side control code (an NPU instruction stream or a C++ TXN
// builder), not aie.core compute: they never go through that backend, and
// aievec ops there are rejected outright ("'aievec.broadcast_scalar' op is
// not supported by the C++ TXN target"). The pass must leave scalar arith
// inside a runtime sequence untouched, regardless of AIE target.
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | FileCheck %s
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s

// CHECK-LABEL: aie.runtime_sequence @clamp_in_runtime_sequence
// CHECK-NOT: aievec.
// CHECK: arith.maxsi
// CHECK: arith.minsi
// CHECK: arith.shrsi
// CHECK-NOT: aievec.
module {
  aie.device(npu2) {
    aie.runtime_sequence @clamp_in_runtime_sequence(%arg0 : i32, %arg1 : i32, %out : memref<i32>) {
      %0 = arith.maxsi %arg0, %arg1 : i32
      memref.store %0, %out[] : memref<i32>
      %1 = arith.minsi %0, %arg1 : i32
      memref.store %1, %out[] : memref<i32>
      %2 = arith.shrsi %1, %arg1 : i32
      memref.store %2, %out[] : memref<i32>
    }
  }
}

// -----

// Companion control: the same scalar ops in an aie.core body (real AIE2
// compute) must still be promoted. The fix for #3549 must not disable the
// optimization outside of runtime sequences.

// CHECK-LABEL: func.func @scalar_ops_in_core
// CHECK: aievec.broadcast_scalar
// CHECK: aievec.broadcast_scalar
// CHECK: aievec.max
// CHECK: aievec.ext_elem
// CHECK-NOT: arith.maxsi
func.func @scalar_ops_in_core(%a: i32, %b: i32) -> i32 {
  %0 = arith.maxsi %a, %b : i32
  return %0 : i32
}
