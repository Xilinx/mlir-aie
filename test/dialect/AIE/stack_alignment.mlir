// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file %s | FileCheck %s

// CHECK: stack_address = 64 : i32
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) { aie.end } { stack_address = 64 : i32 }
  }
}

// -----

// CHECK: stack_address = 32 : i32
module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) { aie.end } { stack_address = 32 : i32 }
  }
}

// -----

// CHECK: stack_address = 32 : i32
module {
  aie.device(xcvc1902) {
    %t = aie.tile(0, 1)
    %c = aie.core(%t) { aie.end } { stack_address = 32 : i32 }
  }
}
