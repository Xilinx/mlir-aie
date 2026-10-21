//===- scratchpad_sync_without_parameters.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A sync in a module that declares no parameters has nothing to stage, so it
// lowers to nothing rather than to a scratchpad of size 0.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-scratchpad-parameters %s | FileCheck %s

// CHECK-LABEL: aie.runtime_sequence @seq
// CHECK-NOT: aiex.npu.create_scratchpad
// CHECK-NOT: aiex.sync_scratchpad_parameters_from_host
// CHECK: }
module {
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    aie.runtime_sequence @seq(%arg0: memref<64xi32>) {
      aiex.sync_scratchpad_parameters_from_host
    }
  }
}
