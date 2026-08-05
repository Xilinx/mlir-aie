//===- bad_stack_size.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file %s 2>&1 | FileCheck %s

// CHECK: error{{.*}}'aie.core' op stack_size 65536 leaves no local memory for this tile's buffers (65536 bytes total)
module @exceeds_local_memory {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %core = aie.core(%t) { aie.end } {stack_size = 65536 : i32}
  }
}

// -----

// CHECK: error{{.*}}'aie.core' op attribute 'stack_size' failed to satisfy constraint
module @zero {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %core = aie.core(%t) { aie.end } {stack_size = 0 : i32}
  }
}

// -----

// CHECK: error{{.*}}'aie.core' op attribute 'stack_size' failed to satisfy constraint
module @negative {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %core = aie.core(%t) { aie.end } {stack_size = -1 : i32}
  }
}
