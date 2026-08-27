//===- odd_reset_count_parity.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-expand-load-pdi --split-input-file %s | FileCheck %s

// The empty-device reset alternates empty_0/empty_1 so that two consecutive
// loads never name the same PDI -- the firmware caches the address and turns a
// repeated load into a no-op. The host re-runs the whole runtime sequence on
// every dispatch, so that alternation has to hold across the dispatch boundary
// too. A sequence with an ODD number of resets would end on the same empty PDI
// it starts with, so the next dispatch's reset silently does nothing and the
// configuration that follows lands on a device that was never reset. One extra
// reset of the opposite parity is appended to keep the sequence even.

// ONE reset -> starts on empty_0, so a trailing empty_1 is appended.
module {
  aie.device(npu2_1col) @dev_a {
    aie.end
  }

  aie.device(npu2_1col) @main {
    // CHECK-LABEL: aie.runtime_sequence(%arg0: memref<1xi32>)
    aie.runtime_sequence (%arg0: memref<1xi32>) {
      // CHECK: aiex.npu.load_pdi {device_ref = @empty_0
      // CHECK: aiex.npu.load_pdi {device_ref = @empty_1
      // CHECK-NOT: aiex.npu.load_pdi
      aiex.npu.load_pdi { device_ref = @dev_a }
    }
  }
}

// -----

// TWO resets already end on the opposite parity, so nothing is appended.
module {
  aie.device(npu2_1col) @dev_a {
    aie.end
  }

  aie.device(npu2_1col) @main {
    // CHECK-LABEL: aie.runtime_sequence(%arg0: memref<1xi32>)
    aie.runtime_sequence (%arg0: memref<1xi32>) {
      // CHECK: aiex.npu.load_pdi {device_ref = @empty_0
      // CHECK: aiex.npu.load_pdi {device_ref = @empty_1
      // CHECK-NOT: aiex.npu.load_pdi
      aiex.npu.load_pdi { device_ref = @dev_a }
      aiex.npu.load_pdi { device_ref = @dev_a }
    }
  }
}
