//===- coverage_unknown_stream_ids.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s

// The second source rule is shadowed. Its phantom receiver at (0,4) must
// not prevent flow 3 from sharing arbiter 0.

// CHECK-LABEL: aie.switchbox(%tile_0_4)
// CHECK:         %[[OLD:.*]] = aie.amsel<0> (0)
// CHECK:         aie.rule(31, 1, %[[OLD]])
// CHECK:         %[[NEW:.*]] = aie.amsel<0> (1)
// CHECK:         aie.rule(31, 3, %[[NEW]])

module {
  aie.device(npu1_1col) {
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    %sb05 = aie.switchbox(%t05) {
      %a0 = aie.amsel<0> (0)
      %a1 = aie.amsel<1> (0)
      aie.masterset(South : 0, %a0)
      aie.masterset(South : 1, %a1)
      aie.packet_rules(DMA : 0) {
        aie.rule(30, 0, %a0)
        aie.rule(31, 1, %a1)
      }
    }
    %sb04 = aie.switchbox(%t04) {
      aie.connect<North : 0, South : 0>
      %a0 = aie.amsel<0> (0)
      aie.masterset(DMA : 1, %a0)
      aie.packet_rules(North : 1) {
        aie.rule(31, 1, %a0)
      }
    }
    %sb03 = aie.switchbox(%t03) {
      aie.connect<North : 0, DMA : 0>
    }
    aie.packet_flow(3) {
      aie.packet_source<%t04, DMA : 0>
      aie.packet_dest<%t03, DMA : 1>
    }
  }
}

// -----

// Both ids pass the source's masked rule. Id 31, not its representative 30,
// reaches (0,4)'s receiver and forces flow 3 onto another arbiter.

// CHECK-LABEL: aie.switchbox(%tile_0_4)
// CHECK:         %[[OLD:.*]] = aie.amsel<0> (0)
// CHECK:         aie.rule(31, 31, %[[OLD]])
// CHECK:         %[[NEW:.*]] = aie.amsel<1> (1)
// CHECK:         aie.rule(31, 3, %[[NEW]])

module {
  aie.device(npu1_1col) {
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    %sb05 = aie.switchbox(%t05) {
      %a0 = aie.amsel<0> (0)
      aie.masterset(South : 0, %a0)
      aie.packet_rules(DMA : 0) {
        aie.rule(30, 30, %a0)
      }
    }
    %sb04 = aie.switchbox(%t04) {
      %a0 = aie.amsel<0> (0)
      %a1 = aie.amsel<1> (0)
      aie.masterset(DMA : 1, %a0)
      aie.masterset(South : 0, %a1)
      aie.packet_rules(North : 0) {
        aie.rule(31, 30, %a1)
        aie.rule(31, 31, %a0)
      }
    }
    %sb03 = aie.switchbox(%t03) {
      aie.connect<North : 0, DMA : 0>
    }
    aie.packet_flow(3) {
      aie.packet_source<%t04, DMA : 0>
      aie.packet_dest<%t03, DMA : 1>
    }
  }
}
