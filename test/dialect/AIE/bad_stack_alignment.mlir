// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: not aie-opt --split-input-file %s 2>&1 | FileCheck %s

// CHECK: 'aie.core' op stack_address must be aligned to 64 bytes for this target's stack ABI
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) { aie.end } { stack_address = 1 : i32 }
  }
}

// -----

// Bus alignment alone is insufficient on AIE2P.
// CHECK: 'aie.core' op stack_address must be aligned to 64 bytes for this target's stack ABI
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) { aie.end } { stack_address = 32 : i32 }
  }
}

// -----

// CHECK: 'aie.core' op stack_address must be aligned to 32 bytes for this target's stack ABI
module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) { aie.end } { stack_address = 16 : i32 }
  }
}

// -----

// AIE1 also requires 32 bytes, rather than its 16-byte vector alignment.
// CHECK: 'aie.core' op stack_address must be aligned to 32 bytes for this target's stack ABI
module {
  aie.device(xcvc1902) {
    %t = aie.tile(0, 1)
    %c = aie.core(%t) { aie.end } { stack_address = 16 : i32 }
  }
}
