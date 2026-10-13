//===- packet_mask_hops.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The rules along one path may state different masks. Each hop narrows the set
// of ids that reach the end, so a lifted flow claims what every hop on its path
// accepts.

// RUN: aie-opt --aie-find-flows --split-input-file %s | FileCheck %s

// The first hop accepts 0x8 through 0xb and the second only 0x9, so the path
// carries 0x9. A full-width mask selects one id, which the id states alone.
// CHECK: aie.packet_flow(9)
// CHECK-NOT: mask

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %sb02 = aie.switchbox(%t02) {
      %a = aie.amsel<0> (0)
      %m = aie.masterset(North : 0, %a)
      aie.packet_rules(DMA : 0) {
        aie.rule(28, 8, %a)
      }
    }
    %sb03 = aie.switchbox(%t03) {
      %a = aie.amsel<0> (0)
      %m = aie.masterset(DMA : 0, %a)
      aie.packet_rules(South : 0) {
        aie.rule(31, 9, %a)
      }
    }
    aie.wire(%t02 : DMA, %sb02 : DMA)
    aie.wire(%sb02 : North, %sb03 : South)
    aie.wire(%t03 : DMA, %sb03 : DMA)
  }
}

// -----

// The two hops accept disjoint sets, so nothing reaches the end. The pass
// recovers no flow and leaves both switchboxes in place, rather than lifting a
// route that carries nothing.

// CHECK-NOT: aie.packet_flow
// CHECK: aie.switchbox
// CHECK:   aie.rule(28, 8
// CHECK: aie.switchbox
// CHECK:   aie.rule(31, 4

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %sb02 = aie.switchbox(%t02) {
      %a = aie.amsel<0> (0)
      %m = aie.masterset(North : 0, %a)
      aie.packet_rules(DMA : 0) {
        aie.rule(28, 8, %a)
      }
    }
    %sb03 = aie.switchbox(%t03) {
      %a = aie.amsel<0> (0)
      %m = aie.masterset(DMA : 0, %a)
      aie.packet_rules(South : 0) {
        aie.rule(31, 4, %a)
      }
    }
    aie.wire(%t02 : DMA, %sb02 : DMA)
    aie.wire(%sb02 : North, %sb03 : South)
    aie.wire(%t03 : DMA, %sb03 : DMA)
  }
}
