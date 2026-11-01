//===- scratchpad_wait_only_sets_lock.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core that waits for the parameters but reads none still gets its lock set
// by the runtime sequence; otherwise its first wait never returns.

// RUN: aie-opt --aie-lower-scratchpad-parameters %s | FileCheck %s

// CHECK: %[[LOCK:.*]] = aie.lock(%{{.*}}) {init = 0 : i32}
// CHECK: aie.use_lock(%[[LOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK: aie.runtime_sequence
// CHECK: aiex.set_lock(%[[LOCK]], %{{.*}})
module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cmax = arith.constant 9223372036854775807 : index
      scf.for %i = %c0 to %cmax step %c1 {
        aiex.wait_scratchpad_parameters
      }
      aie.end
    }
    aie.runtime_sequence(%arg0 : memref<16xi32>) {
    }
  }
}
