//===- deterministic_merge.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: sed 's/NPUDEVICE/npu1/g' %s | aie-translate --aie-generate-xaie | FileCheck %s
// RUN: sed 's/NPUDEVICE/npu2/g' %s | aie-translate --aie-generate-xaie | FileCheck %s

// The `deterministic_merge` schedule on an amsel must reach the libxaie
// configuration. This backend previously dropped it silently, which is the
// worst failure mode for the feature: the design compiles and runs, but with
// free arbitration and no diagnostic.
//
// Slots are emitted in array order (position 0, 1, ...), and the enable must
// come last -- writing it resets the feature's internal state, so it has to
// follow the slot configuration.
//
// Run for both parts: deterministic merge is an AIE2 property, and npu1 and
// npu2 both expose two arbiters with four slots each, so the same schedule must
// verify and emit on either.
//
// See test/npu-xrt/packet_flow_merge_order for the hardware test proving the
// arbiter actually honours a programmed order.

// CHECK: XAie_StrmSwDeterministicMergeConfig({{.*}}, /* arbiter */ 0, NORTH, 1, /* pkt_count */ 1, /* position */ 0)
// CHECK-NEXT: XAie_StrmSwDeterministicMergeConfig({{.*}}, /* arbiter */ 0, DMA, 0, /* pkt_count */ 3, /* position */ 1)
// CHECK-NEXT: XAie_StrmSwDeterministicMergeEnable({{.*}}, /* arbiter */ 0)

module {
  aie.device(NPUDEVICE) {
    %t = aie.tile(0, 3)
    aie.switchbox(%t) {
      %a0 = aie.amsel<0> (0) deterministic_merge [<North : 1, 1>, <DMA : 0, 3>]
      aie.masterset(South : 1, %a0)
      aie.packet_rules(North : 1) { aie.rule(31, 2, %a0) }
      aie.packet_rules(DMA : 0) { aie.rule(31, 1, %a0) }
    }
  }
}
