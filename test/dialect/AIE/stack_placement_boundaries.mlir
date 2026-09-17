// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file %s | FileCheck %s

// An explicitly pinned stack may end exactly at its bank boundary.
// CHECK: stack_address = 31744 : i32, stack_bank = 1 : i32
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) { aie.end } { stack_size = 1024 : i32, stack_address = 31744 : i32, stack_bank = 1 : i32 }
  }
}

// -----

// An address-only pin may end exactly at its bank boundary.
// CHECK: stack_address = 31744 : i32, stack_size = 1024 : i32
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) { aie.end } { stack_size = 1024 : i32, stack_address = 31744 : i32 }
  }

  // -----

  // Preserve legacy placement for a large stack with no placement hints.
  // CHECK: stack_size = 32768 : i32
  module {
    aie.device(npu2) {
      %t = aie.tile(0, 2)
      %c = aie.core(%t) { aie.end } { stack_size = 32768 : i32 }
    }
  }
}

// -----

// Ending exactly at the tile boundary is legal.
// CHECK: stack_address = 64512 : i32, stack_bank = 3 : i32
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) { aie.end } { stack_size = 1024 : i32, stack_address = 64512 : i32, stack_bank = 3 : i32 }
  }
}
