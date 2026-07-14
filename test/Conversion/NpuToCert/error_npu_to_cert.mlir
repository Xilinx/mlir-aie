//===- error_npu_to_cert.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-npu-to-cert %s 2>&1 | FileCheck %s

// Test: aiex.run argument count does not match the callee runtime sequence's

// CHECK: error: number of run op arguments (1) does not match number of callee runtime sequence arguments (0)

module {
  aie.device(npu2) {
    aie.runtime_sequence @configure(%x: memref<1xi32>) {
      aiex.configure @callee {
        aiex.run @run_a(%x) : (memref<1xi32>)
      }
    }
  }
  aie.device(npu2) @callee {
    aie.runtime_sequence @run_a() {
    }
  }
}
