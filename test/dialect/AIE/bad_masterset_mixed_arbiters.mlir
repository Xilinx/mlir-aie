//===- bad_masterset_mixed_arbiters.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}} 'aie.masterset' op a master port can only be tied to one arbiter

// A master port folds down to a single arbiter plus an msel mask, so amsels
// straddling arbiters have no valid lowering.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      %a0 = aie.amsel<0> (0)
      %a1 = aie.amsel<1> (1)
      aie.masterset(North : 0, %a0, %a1)
      aie.packet_rules(South : 0) { aie.rule(31, 0, %a0) }
      aie.packet_rules(South : 1) { aie.rule(31, 1, %a1) }
    }
  }
}
