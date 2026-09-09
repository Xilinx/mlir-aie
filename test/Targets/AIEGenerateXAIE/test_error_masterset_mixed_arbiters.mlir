//===- test_error_masterset_mixed_arbiters.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-translate --aie-generate-xaie %s 2>&1 | FileCheck %s
// CHECK: error: 'aie.masterset' op a master port can only be tied to one arbiter

// Use a logical tile so SwitchboxOp verifier defers this check; translation
// must still reject the malformed masterset instead of silently lowering it.
module @test_error_masterset_mixed_arbiters {
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
