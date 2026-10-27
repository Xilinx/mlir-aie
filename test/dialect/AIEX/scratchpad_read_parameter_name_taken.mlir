//===- scratchpad_read_parameter_name_taken.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A parameter's buffer skips a name a buffer already has.

// RUN: aie-opt --aie-lower-scratchpad-parameters %s | FileCheck %s

// CHECK-DAG: aie.buffer @__param_n_0_2_0(%{{.*}}) : memref<4xi32>
// CHECK-DAG: aie.buffer @__param_n_0_2_1(%{{.*}}) : memref<2xi32>
module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    aie.buffer @__param_n_0_2_0(%t02) : memref<4xi32>
    aie.core(%t02) {
      %v = aiex.read_scratchpad_parameter @n : i32
      aie.end
    }
  }
}
