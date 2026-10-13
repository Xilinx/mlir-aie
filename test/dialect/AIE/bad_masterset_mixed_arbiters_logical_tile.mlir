//===- bad_masterset_mixed_arbiters_logical_tile.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}} 'aie.masterset' op a master port can only be tied to one arbiter

// SwitchboxOp::verify defers its target-model checks until after
// --aie-place-tiles, so on an aie.logical_tile the single-arbiter invariant has
// to come from MasterSetOp::verify. Without it this IR reaches the
// stream-switch backends, which would silently program the master port for
// whichever arbiter came last.
module {
  aie.device(npu2) {
    %t = aie.logical_tile<CoreTile>(0, 2)
    aie.switchbox(%t) {
      %a0 = aie.amsel<0> (0)
      %a1 = aie.amsel<1> (1)
      aie.masterset(North : 0, %a0, %a1)
      aie.end
    }
  }
}
