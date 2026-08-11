//===- deterministic_merge.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | aie-opt | FileCheck %s

// Round-trip for the `deterministic_merge` schedule carried on aie.amsel.
//
// The schedule puts the amsel's *arbiter* into deterministic merge mode: the
// arbiter grants slot 0's slave port `packet_count` times, then slot 1's, and so
// on, instead of arbitrating freely. See AMSelOp's description, and
// test/npu-xrt/packet_flow_merge_order for a design that proves on real
// hardware that the arbiter honours the programmed order.

// CHECK-LABEL: aie.device(npu1)
// CHECK:       %[[a0:.*]] = aie.amsel<0> (0) deterministic_merge [<North : 1, 1>, <DMA : 0, 1>]
// CHECK:       aie.masterset(South : 1, %[[a0]])

module {
  aie.device(npu1) {
    %t = aie.tile(0, 3)
    aie.switchbox(%t) {
      %a0 = aie.amsel<0> (0) deterministic_merge [<North : 1, 1>, <DMA : 0, 1>]
      aie.masterset(South : 1, %a0)
      aie.packet_rules(North : 1) { aie.rule(31, 2, %a0) }
      aie.packet_rules(DMA : 0) { aie.rule(31, 1, %a0) }
    }
  }
}

// -----

// The Figure 3-16(b) shape from the architecture spec: a slave may hold more
// than one slot, and a slot may grant several packets in a row. Here North : 1
// is served once, then DMA : 0, then North : 1 three more times, then West : 3.
//
// A repeated slave is why the schedule cannot simply be derived from the
// packet_rules -- the rules say which slaves feed the arbiter, but not how many
// times or in what order each is served.

// CHECK-LABEL: aie.device(npu1)
// CHECK:       aie.amsel<0> (0) deterministic_merge [<North : 1, 1>, <DMA : 0, 1>, <North : 1, 3>, <West : 3, 1>]
module {
  aie.device(npu1) {
    %t = aie.tile(1, 3)
    aie.switchbox(%t) {
      %a0 = aie.amsel<0> (0) deterministic_merge [
              <North : 1, 1>, <DMA : 0, 1>, <North : 1, 3>, <West : 3, 1>]
      aie.masterset(South : 1, %a0)
      aie.packet_rules(North : 1) { aie.rule(31, 2, %a0) }
      aie.packet_rules(DMA : 0) { aie.rule(31, 1, %a0) }
      aie.packet_rules(West : 3) { aie.rule(31, 4, %a0) }
    }
  }
}
