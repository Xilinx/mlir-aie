//===- scratchpad_read_parameter_lowering.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A core read of a scratchpad parameter lowers to a load from its buffer and a
// shift right by 2, even when the input uses no memref op of its own.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-scratchpad-parameters %s | FileCheck %s

// CHECK: %[[BUF:.*]] = aie.buffer(%{{.*}}) {sym_name = "__param_n_0_2_{{[0-9]+}}"} : memref<2xi32>
// CHECK: aie.core
// CHECK: %[[RAW:.*]] = memref.load %[[BUF]][%{{.*}}] : memref<2xi32>
// CHECK: arith.shrui %[[RAW]], %{{.*}} : i32
module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    aie.core(%t02) {
      %v = aiex.read_scratchpad_parameter @n : i32
      aie.end
    }
  }
}
