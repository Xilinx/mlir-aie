//===- unreachable_dest_err_test.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A flow whose destination cannot be reached by the router exercises the
// trace-back unreachable-destination guard (no predecessor for the dest), which
// must fail the routing cleanly rather than indexing with a -1 predecessor.

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s
// CHECK: error: Unable to find a legal routing: no path leads from (0, 5) DMA:0 to (0, 5) DMA:1

// A core tile's DMA:0 cannot drive its own DMA:1 through the switchbox, and
// the top tile of a one-column device can leave only south, where existing
// master sets hold every port.
module {
  aie.device(npu1_1col) {
    %t05 = aie.tile(0, 5)
    %sb05 = aie.switchbox(%t05) {
      %a0 = aie.amsel<0> (0)
      aie.masterset(South : 0, %a0)
      aie.masterset(South : 1, %a0)
      aie.masterset(South : 2, %a0)
      aie.masterset(South : 3, %a0)
    }
    aie.flow(%t05, DMA : 0, %t05, DMA : 1)
  }
}
