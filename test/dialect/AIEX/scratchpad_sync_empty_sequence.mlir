//===- scratchpad_sync_empty_sequence.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An empty runtime sequence parses to a region with no block. The core still
// waits on the parameter-sync lock, so the sequence still gets the preamble.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-scratchpad-parameters %s | FileCheck %s

// CHECK-LABEL: aie.runtime_sequence @empty
// CHECK-NEXT: aiex.npu.create_scratchpad {size = 4 : ui32}
// CHECK: aiex.npu.update_from_scratchpad
// CHECK: aiex.set_lock
module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    aie.core(%t02) {
      %v = aiex.read_scratchpad_parameter @n : i32
      aie.end
    }
    aie.runtime_sequence @empty() {
    }
  }
}
