//===- coverage_switchbox_verify_order.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A slave port driving both a connect and packet_rules is rejected whichever
// op comes first.

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      %a0 = aie.amsel<0> (0)
      %m = aie.masterset(South : 0, %a0)
      // expected-error@+1 {{packet switched source DMA0 cannot match another connect or masterset operation}}
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 3, %a0)
      }
      aie.connect<DMA : 0, North : 0>
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %sb02 = aie.switchbox(%t02) {
      aie.connect<DMA : 0, North : 0>
      %a0 = aie.amsel<0> (0)
      %m = aie.masterset(South : 0, %a0)
      // expected-error@+1 {{packet switched source DMA0 cannot match another connect or masterset operation}}
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 3, %a0)
      }
    }
  }
}
