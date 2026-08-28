//===- program_memory_reserved_invalid.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Reservations that cannot mean anything are rejected at the op, where the
// number is written, rather than surfacing later as a confusing link error.

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: program_memory_reserved 16384 leaves no program memory for this core's code (16384 bytes total)
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) {
      aie.end
    } { program_memory_reserved = 16384 : i32 }
  }
}

// -----

// A run-time write into the reservation has to start on a whole program-memory
// line, so a reservation that does not begin on one is unusable for the thing
// it exists for.
// CHECK: program_memory_reserved 8200 is not a multiple of the 16-byte program-memory line
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) {
      aie.end
    } { program_memory_reserved = 8200 : i32 }
  }
}
