//===- bad_stack_placement.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A stack pin the allocator could never honor is a user constraint, so it is
// rejected here rather than at placement.

// RUN: not aie-opt --split-input-file %s 2>&1 | FileCheck %s

// CHECK: error{{.*}}'aie.core' op stack_bank 9 does not exist; this tile has 4 banks
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) { aie.end } { stack_size = 1024 : i32, stack_bank = 9 : i32 }
  }
}

// -----

// Starting in the requested bank is insufficient: the entire stack must fit.
// CHECK: error{{.*}}'aie.core' op a 1024-byte stack at 0x7E00 runs past stack_bank 1 (ending at 0x8000)
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) { aie.end } { stack_size = 1024 : i32, stack_address = 32256 : i32, stack_bank = 1 : i32 }
  }
}

// -----

// A stack larger than a bank cannot sit in one, and pinning it to a bank is
// then a contradiction rather than a placement problem.
// CHECK: error{{.*}}'aie.core' op stack_bank pins a 32768-byte stack to bank 1, which holds 16384 bytes; omit stack_bank and stack_address for legacy placement
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) { aie.end } { stack_size = 32768 : i32, stack_bank = 1 : i32 }
  }
}

// -----

// Both given and disagreeing: one of the two is stale.
// CHECK: error{{.*}}'aie.core' op stack_address 0x400 lies in bank 0, but stack_bank requests bank 2
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) { aie.end } { stack_size = 1024 : i32, stack_address = 1024 : i32, stack_bank = 2 : i32 }
  }
}

// -----

// Fits the tile, but placed so it runs off the end.
// CHECK: error{{.*}}'aie.core' op a 1024-byte stack at 0xFE00 runs past this tile's local memory (65536 bytes total)
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) { aie.end } { stack_size = 1024 : i32, stack_address = 65024 : i32 }
  }
}
